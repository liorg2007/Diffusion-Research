"""Phase 3 baseline evaluation (D-009 protocol). Single entry point; every stage is resumable.

  python -I experiments/003-baselines-eval/run.py prep            # preprocess + splits + assertions
  python -I experiments/003-baselines-eval/run.py score [names]   # 4 pretrained scorers x 3 conditions (MPS + CPU guard)
  python -I experiments/003-baselines-eval/run.py b0              # DINOv2-base + logistic probe (+ LOGO)
  python -I experiments/003-baselines-eval/run.py report          # metrics, results.json, tables.md
  python -I experiments/003-baselines-eval/run.py all

Caches live in experiments/003-baselines-eval/cache/ (git-ignored). Seed 0 everywhere.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from PIL import Image

CACHE = HERE / "cache"
PROC = ROOT / "data" / "processed" / "crop512"
SEED = 0
CONDITIONS = ["clean", "jpeg75_resize05", "social_sim"]
SCORERS = {  # name -> (module, function)
    "communityforensics": ("scorers.communityforensics", "score"),
    "bfree": ("scorers.bfree", "score"),
    "rine": ("scorers.rine", "score"),
    "clipdet": ("scorers.clipdet", "score"),
}
B0 = "b0_dinov2"
DINO_SPEC = "hf:dinov2:facebook/dinov2-base"
FAKE_GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-xl", "stable-diffusion-2",
             "sd3_large", "flux"]
POOLS_REAL = ["vision_native", "raise", "vision_whatsapp", "vision_facebook_hq"]
MPS_BATCH = 20          # D-008
TOL = 1e-3              # D-008 abort threshold on max |dlogit| MPS vs CPU
CALIB_FRAC = 0.30
N_TRAIN_DEV = 10
CFG = dict(seed=SEED, crop=512, conditions=CONDITIONS, scorers={k: f"{m}:{f}" for k, (m, f) in SCORERS.items()},
           b0=dict(backbone=DINO_SPEC, input_size=224, layers=[-1], use_cls=True, use_mean=True, probe="probe.fit_probe"),
           mps_batch=MPS_BATCH, d008_tol=TOL, d008_subset=32, calib_frac=CALIB_FRAC, fpr_calib=0.05,
           n_train_devices=N_TRAIN_DEV, n_boot=1000, threshold_native=0.0,
           excluded_generators=["latent-diffusion"])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(o, p):
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    Path(p).write_text(json.dumps(conv(o), indent=1))


# ----------------------------------------------------------------------------- prep
def _proc_one(a):
    from preprocess import process_image
    src, dst = a
    return process_image(src, dst, 512)


def stage_prep(force=False):
    from manifest import write_manifest
    CACHE.mkdir(exist_ok=True)
    (HERE / ".gitignore").write_text("cache/\n")
    if (HERE / "splits.csv").exists() and (PROC / "sidecar.csv").exists() and not force:
        log("prep: already done (use --force to redo)")
        return
    raw = pd.read_csv("data/manifest_raw.csv", keep_default_na=False)
    rep = dict(n_raw=len(raw))
    raw["split"] = ""
    # exclusions
    ldm = raw.generator == "latent-diffusion"
    rep["excluded_latent_diffusion_256px"] = int(ldm.sum())
    raw = raw[~ldm]
    dup = raw.sha256.duplicated(keep="first")
    rep["skipped_duplicate_sha256"] = int(dup.sum())
    rep["duplicate_paths"] = raw.path[dup].tolist()
    raw = raw[~dup].reset_index(drop=True)
    # preprocess (format detected from content by PIL inside process_image)
    (PROC / "images").mkdir(parents=True, exist_ok=True)
    t = time.time()
    import multiprocessing as mp
    jobs = [(r.path, str(PROC / "images" / f"{r.sha256}.png")) for r in raw.itertuples()]
    with mp.Pool(4) as pool:
        out = pool.map(_proc_one, jobs, chunksize=16)
    side = []
    for r, (status, info) in zip(raw.itertuples(), out):
        side.append(dict(sha256=r.sha256, src_path=r.path, label=r.label, generator=r.generator, status=status,
                         crop_size=512, **info))
    side = pd.DataFrame(side)
    side.to_csv(PROC / "sidecar.csv", index=False)
    rep.update(n_ok=int((side.status == "ok").sum()), n_skipped_small=int((side.status == "skipped_small").sum()),
               n_error=int((side.status == "error").sum()), preprocess_seconds=time.time() - t)
    bad = side[side.status != "ok"]
    rep["skipped_other_detail"] = bad.groupby(["generator", "status"]).size().rename("n").reset_index().to_dict("records")
    rep["orig_format_by_generator"] = (side.groupby(["generator", "orig_format"]).size().rename("n").reset_index()
                                       .to_dict("records"))
    ok = set(side.sha256[side.status == "ok"])
    df = raw[raw.sha256.isin(ok)].copy().reset_index(drop=True)
    df["src_path"] = df["path"]
    df["path"] = [str(PROC / "images" / f"{s}.png") for s in df.sha256]
    # real file format from content (sidecar), never extension
    df["orig_format"] = side.set_index("sha256").loc[df.sha256, "orig_format"].values
    fire = df[df.generator == "firefly"]
    rep["firefly_orig_format_counts"] = fire.orig_format.value_counts().to_dict()
    assert (fire.orig_format == "JPEG").all(), "firefly should be JPEG-in-.png"
    # pools / devices
    pool, dev, vidx = [], [], []
    for r in df.itertuples():
        if r.generator != "real":
            pool.append("fake"); dev.append(""); vidx.append("")
        elif r.camera_or_platform == "RAISE":
            pool.append("raise"); dev.append(""); vidx.append("")
        else:
            m = re.fullmatch(r"(D\d+)_(native|whatsapp|facebook_hq)", r.camera_or_platform)
            assert m, r.camera_or_platform
            pool.append({"native": "vision_native", "whatsapp": "vision_whatsapp",
                         "facebook_hq": "vision_facebook_hq"}[m.group(2)])
            dev.append(m.group(1))
            vidx.append(re.search(r"_(\d{4})\.jpg$", r.src_path).group(1))
    df["pool"], df["device"], df["vidx"] = pool, dev, vidx
    # ---- splits (D-009)
    rng = np.random.default_rng(SEED)
    split = pd.Series("", index=df.index, dtype=object)
    devices = sorted(set(d for d in df.device if d))
    assert len(devices) == 35
    train_dev = sorted(rng.choice(devices, N_TRAIN_DEV, replace=False).tolist())
    test_dev = [d for d in devices if d not in train_dev]
    # VISION: triplets (device, vidx); WA/FBH follow native
    nat = df[df.pool == "vision_native"]
    trip_split = {}
    for d in test_dev:
        for v in nat.vidx[nat.device == d]:
            trip_split[(d, v)] = "test"
    for d in train_dev:
        for v in sorted(nat.vidx[nat.device == d]):
            trip_split[(d, v)] = "train"
    train_trips = sorted(k for k, s in trip_split.items() if s == "train")
    ncal = int(round(CALIB_FRAC * len(train_trips)))
    for i in rng.permutation(len(train_trips))[:ncal]:
        trip_split[train_trips[i]] = "calib"
    for i in df.index[df.device != ""]:
        split[i] = trip_split[(df.device[i], df.vidx[i])]
    # RAISE 50/50 by image, calib = CALIB_FRAC of the train half
    ri = np.array(sorted(df.index[df.pool == "raise"], key=lambda i: df.sha256[i]))
    rp = rng.permutation(len(ri))
    nt = len(ri) // 2
    rtest, rtrain = ri[rp[:nt]], ri[rp[nt:]]
    split[rtest] = "test"
    split[rtrain] = "train"
    ncr = int(round(CALIB_FRAC * len(rtrain)))
    split[rtrain[rng.permutation(len(rtrain))[:ncr]]] = "calib"
    # fakes
    for g in sorted(set(df.generator[df.label == 1])):
        gi = np.array(sorted(df.index[(df.label == 1) & (df.generator == g)], key=lambda i: df.sha256[i]))
        if g == "flux":
            split[gi] = "test"
        else:
            p = rng.permutation(len(gi))
            h = len(gi) // 2
            split[gi[p[:h]]] = "test"
            split[gi[p[h:]]] = "train"
    df["split"] = split
    # ---- assertions
    assert (df.split != "").all()
    v = df[df.device != ""]
    dsp = v.groupby("device").split.agg(lambda s: set(s))
    for d, s in dsp.items():
        assert (d in test_dev and s == {"test"}) or (d in train_dev and s <= {"train", "calib"}), (d, s)
    assert not (set(v.device[v.split == "test"]) & set(v.device[v.split.isin(["train", "calib"])])), "device in both sides"
    assert (df[df.generator == "flux"].split == "test").all() and len(df[df.generator == "flux"]) > 0, "FLUX leaked"
    assert not df.sha256.duplicated().any()
    assert (v.groupby(["device", "vidx"]).split.nunique() == 1).all(), "VISION triplet crosses splits"
    assert (df[df.label == 1].split.isin(["train", "test"])).all()
    assert (df[df.split == "calib"].label == 0).all()
    df["in_scoring_set"] = (df.split == "test") | ((df.split == "calib") & df.pool.isin(["vision_native", "raise"]))
    df["b0_train"] = (df.split == "train") & df.pool.isin(["vision_native", "raise", "fake"])
    cols = ["sha256", "path", "src_path", "label", "generator", "pool", "camera_or_platform", "device", "vidx", "split",
            "in_scoring_set", "b0_train", "orig_format", "source_dataset", "license"]
    df[cols].to_csv(HERE / "splits.csv", index=False)
    write_manifest(df.drop(columns=["src_path", "pool", "device", "vidx", "in_scoring_set", "b0_train"], errors="ignore"),
                   PROC / "manifest_processed.csv")
    rep.update(train_devices=train_dev, test_devices=test_dev,
               split_counts={f"{s}/{p}/{g}": int(n) for (s, p, g), n in
                             df.groupby(["split", "pool", "generator"]).size().items()},
               n_processed=len(df))
    jdump(rep, CACHE / "prep_report.json")
    log("prep done", json.dumps({k: rep[k] for k in ["n_raw", "n_ok", "n_skipped_small", "n_error",
                                                    "excluded_latent_diffusion_256px", "skipped_duplicate_sha256"]}))
    log("\n" + df.groupby(["split", "pool"]).size().to_string())


def load_splits():
    return pd.read_csv(HERE / "splits.csv", keep_default_na=False, dtype={"vidx": str})


def check_subset(df):
    p = CACHE / "check_subset.json"
    if p.exists():
        return json.loads(p.read_text())
    s = df[df.in_scoring_set]
    rng = np.random.default_rng(SEED)
    r = sorted(s.sha256[s.label == 0]); f = sorted(s.sha256[s.label == 1])
    sub = [r[i] for i in rng.choice(len(r), 16, replace=False)] + [f[i] for i in rng.choice(len(f), 16, replace=False)]
    p.write_text(json.dumps(sub))
    return sub


def load_cond(path, sha, cond):
    from perturb import apply_condition
    with Image.open(path) as im:
        im.load()
        return apply_condition(im, cond, key=sha, seed=SEED)


# ----------------------------------------------------------------------------- score
def _get_fn(name):
    import importlib
    m, f = SCORERS[name]
    return getattr(importlib.import_module(m), f)


def _score_rows(fn, rows, cond, batch=MPS_BATCH):
    out = {}
    for i in range(0, len(rows), batch):
        ch = rows.iloc[i:i + batch]
        imgs = [load_cond(p, s, cond) for p, s in zip(ch.path, ch.sha256)]
        sc = np.asarray(fn(imgs), float).reshape(-1)
        assert len(sc) == len(imgs) and np.isfinite(sc).all()
        out.update(zip(ch.sha256, sc))
    return out


def _load_scores(name, cond):
    p = CACHE / f"scores_{name}_{cond}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return dict(zip(z["keys"].tolist(), z["vals"].tolist()))
    return {}


def _save_scores(name, cond, d):
    p = CACHE / f"scores_{name}_{cond}.npz"
    ks = sorted(d)
    np.savez(str(p) + ".tmp.npz", keys=np.array(ks), vals=np.array([d[k] for k in ks]))
    os.replace(str(p) + ".tmp.npz", p)


def stage_cpuref(name):
    """Subprocess entry: CPU logits for the fixed 32-image subset under every condition."""
    os.environ["DD_DEVICE"] = "cpu"
    dfl = load_splits()
    sub = check_subset(dfl)
    df = dfl.set_index("sha256", drop=False)
    fn = _get_fn(name)
    p = CACHE / f"cpuref_{name}.json"
    ref = json.loads(p.read_text()) if p.exists() else {}
    for cond in CONDITIONS:
        if cond in ref:
            continue
        t = time.time()
        ref[cond] = _score_rows(fn, df.loc[sub], cond, batch=4)
        p.write_text(json.dumps(ref))
        log(f"cpuref {name} {cond} done in {time.time() - t:.0f}s")


def stage_score(names=None):
    df = load_splits()
    dfi = df.set_index("sha256", drop=False)
    sub = check_subset(df)
    rest_rows = df[df.in_scoring_set & ~df.sha256.isin(sub)]
    tim_p = CACHE / "timings.json"
    timings = json.loads(tim_p.read_text()) if tim_p.exists() else {}
    guard_p = CACHE / "d008_guard.json"
    guard = json.loads(guard_p.read_text()) if guard_p.exists() else {}
    for name in names or SCORERS:
        rp = CACHE / f"cpuref_{name}.json"
        if not rp.exists() or any(c not in json.loads(rp.read_text()) for c in CONDITIONS):
            log(f"{name}: computing CPU reference ({len(sub)} imgs x {len(CONDITIONS)} conds)")
            r = subprocess.run([sys.executable, "-I", str(HERE / "run.py"), "cpuref", name], check=False)
            if r.returncode != 0:
                raise SystemExit(f"cpuref failed for {name}")
        ref = json.loads(rp.read_text())
        os.environ["DD_DEVICE"] = "mps"
        fn = _get_fn(name)
        for cond in CONDITIONS:
            t0 = time.time()
            sc = _load_scores(name, cond)
            todo_sub = dfi.loc[[s for s in sub if s not in sc]]
            if len(todo_sub):
                sc.update(_score_rows(fn, todo_sub, cond))
                _save_scores(name, cond, sc)
            d = np.array([sc[s] - ref[cond][s] for s in sub])
            mx = float(np.abs(d).max())
            guard[f"{name}/{cond}"] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL))
            guard_p.write_text(json.dumps(guard, indent=1))
            log(f"{name} {cond}: D-008 check max|dlogit|={mx:.2e} (tol {TOL})")
            if mx > TOL:
                raise SystemExit(f"D-008 ABORT: {name}/{cond} max |dlogit| {mx:.3e} > {TOL}")
            todo = rest_rows[~rest_rows.sha256.isin(sc)]
            for i in range(0, len(todo), 100):
                sc.update(_score_rows(fn, todo.iloc[i:i + 100], cond))
                _save_scores(name, cond, sc)
                log(f"{name} {cond}: {len(sc)}/{len(rest_rows) + len(sub)}")
            timings[f"{name}/{cond}"] = timings.get(f"{name}/{cond}", 0) + time.time() - t0
            tim_p.write_text(json.dumps(timings, indent=1))


# ----------------------------------------------------------------------------- B0
def stage_b0():
    import probe
    from features import extract_features, load_backbone
    df = load_splits()
    sub = check_subset(df)
    bb = load_backbone(DINO_SPEC, input_size=224, layers=(-1,), use_cls=True, use_mean=True)
    t0 = time.time()
    in_train_eval = (df.label == 1) & (df.split == "train")            # for LOGO (train-split fakes of the left-out gen)
    X = {}
    for cond in CONDITIONS:
        m = df.in_scoring_set | in_train_eval
        if cond == "clean":
            m = m | df.b0_train
        idx = np.where(m.to_numpy())[0]
        # D-010: DINOv2's bicubic pos-embed interpolation is unsupported on MPS (torch 2.2.2); run on CPU only
        f = extract_features(bb, df.path.iloc[idx], cond, str(CACHE / "b0"), 16, keys=df.sha256.iloc[idx], seed=SEED,
                             device="cpu")
        Xc = np.zeros((len(df), f.shape[1]), np.float32)
        Xc[idx] = f
        X[cond] = Xc
        log(f"b0 features {cond}: {len(idx)} imgs, dim {f.shape[1]}")
    t_feat = time.time() - t0
    tr = df.b0_train.to_numpy()
    y = df.label.to_numpy()
    model = probe.fit_probe(X["clean"][tr], y[tr], SEED)
    chosen_C = float(model[-1].C_[0])
    log(f"b0 probe trained n={tr.sum()} C={chosen_C}")
    # D-008 CPU guard on the decision logit
    rng = np.random.default_rng(SEED)
    trn = df.sha256[tr].to_numpy()
    trsub = [str(trn[i]) for i in rng.choice(len(trn), 16, replace=False)]
    dfi = df.set_index("sha256", drop=False)
    order = {s: i for i, s in enumerate(df.sha256)}
    guard = {}
    for cond in CONDITIONS:
        keys = sub + (trsub if cond == "clean" else [])
        rows = dfi.loc[keys]
        Xcpu = extract_features(bb, rows.path, cond, str(CACHE / "b0_cpu"), 4, keys=rows.sha256, seed=SEED, device="cpu")
        Xm = X[cond][[order[s] for s in keys]]
        mx = float(np.abs(model.decision_function(Xm) - model.decision_function(Xcpu)).max())
        guard[f"{B0}/{cond}"] = dict(max_abs_dlogit=mx, n=len(keys), tol=TOL, ok=bool(mx <= TOL),
                                     max_abs_feature_diff=float(np.abs(Xm - Xcpu).max()))
        log(f"{B0} {cond}: D-008 check max|dlogit|={mx:.2e}, max|dfeat|={guard[f'{B0}/{cond}']['max_abs_feature_diff']:.2e}")
        if mx > TOL:
            raise SystemExit(f"D-008 ABORT: {B0}/{cond} max |dlogit| {mx:.3e} > {TOL}")
    gp = CACHE / "d008_guard.json"
    g0 = json.loads(gp.read_text()) if gp.exists() else {}
    g0.update(guard)
    gp.write_text(json.dumps(g0, indent=1))
    for cond in CONDITIONS:
        m = df.in_scoring_set.to_numpy()
        _save_scores(B0, cond, dict(zip(df.sha256[m], model.decision_function(X[cond][m]))))
    # LOGO over the 7 non-FLUX generators (rows: native + RAISE reals + fakes, as in B0 training)
    keep = df.pool.isin(["vision_native", "raise", "fake"]).to_numpy()
    dsub = df[keep].reset_index(drop=True)
    logo = {}
    for cond in CONDITIONS:
        r = probe.logo_eval(dsub, X["clean"][keep], X[cond][keep], SEED, 0.5, 1000,
                            generators=[g for g in FAKE_GENS if g != "flux"])
        logo[cond] = r
        log(f"b0 LOGO {cond}: macro AUROC {r['macro']['auroc']:.4f}")
    jdump(dict(chosen_C=chosen_C, n_train=int(tr.sum()), n_train_real=int((y[tr] == 0).sum()),
               feature_seconds=t_feat, logo=logo), CACHE / "b0_results.json")
    tp = CACHE / "timings.json"
    tm = json.loads(tp.read_text()) if tp.exists() else {}
    tm[f"{B0}/features+probe+logo"] = time.time() - t0
    tp.write_text(json.dumps(tm, indent=1))


# ----------------------------------------------------------------------------- report
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0, c - h)), float(min(1, c + h)))


def _eval_one(args):
    import metrics
    name, cond, rows, s_test, s_calib, clean_calib = args
    out = {}
    lab = rows.label.to_numpy(); pool = rows.pool.to_numpy()
    main = ((pool == "vision_native") | (pool == "raise") | (lab == 1))
    M = rows[main].reset_index(drop=True)
    sM = s_test[main]
    r = metrics.evaluate_scored_frame(M, sM, s_calib, 0.0, 1000, SEED)
    out["overall"], out["per_generator"] = r["overall"], r["per_generator"]
    nf = (M.generator != "flux").to_numpy()
    out["overall_noflux"] = metrics.compute_metrics(M.label.to_numpy()[nf], sM[nf], 0.0, s_calib, 1000, SEED)
    out["per_real_pool"] = {}
    for rp in ["vision_native", "raise"]:
        m = ((M.pool == rp) | (M.label == 1)).to_numpy()
        out["per_real_pool"][rp] = metrics.compute_metrics(M.label.to_numpy()[m], sM[m], 0.0, s_calib, 500, SEED)
    thr = metrics.threshold_for_fpr(s_calib, 0.05)
    thr_clean = metrics.threshold_for_fpr(clean_calib, 0.05)
    fs = {}
    for rp in POOLS_REAL:
        v = s_test[(pool == rp) & (lab == 0)]
        n = len(v)
        e = {}
        for tag, t in [("calib5", thr), ("native0", 0.0), ("clean_calib5", thr_clean)]:
            k = int((v >= t).sum())
            e[tag] = dict(fpr=k / n if n else np.nan, n=n, k=k, ci=wilson(k, n))
        e["median_logit"] = float(np.median(v)) if n else np.nan
        fs[rp] = e
    out["fpr_by_source"] = fs
    out["thresholds"] = dict(calib5=float(thr), clean_calib5=float(thr_clean), native=0.0)
    out["fpr_calib_self"] = float((s_calib >= thr).mean())
    fk = s_test[lab == 1]
    out["tpr_at_clean_calib5"] = float((fk >= thr_clean).mean())
    out["tpr_at_calib5_allfakes"] = float((fk >= thr).mean())
    out["tpr_at_native0_allfakes"] = float((fk >= 0).mean())
    return name, cond, out


def stage_report():
    import concurrent.futures as cf
    import hashlib
    import format_bias_probe
    df = load_splits()
    side = pd.read_csv(PROC / "sidecar.csv")
    test = df[df.split == "test"].reset_index(drop=True)
    calib = df[(df.split == "calib") & df.pool.isin(["vision_native", "raise"])].reset_index(drop=True)
    names = list(SCORERS) + [B0]
    tasks = []
    for name in names:
        clean_c = np.array([_load_scores(name, "clean")[s] for s in calib.sha256])
        for cond in CONDITIONS:
            sc = _load_scores(name, cond)
            tasks.append((name, cond, test, np.array([sc[s] for s in test.sha256]),
                          np.array([sc[s] for s in calib.sha256]), clean_c))
    t = time.time()
    res = {n: {} for n in names}
    with cf.ProcessPoolExecutor(4) as ex:
        for name, cond, out in ex.map(_eval_one, tasks):
            res[name][cond] = out
    log(f"metrics done in {time.time() - t:.0f}s")
    fb = {}
    for tag, pools in [("vision_native_vs_fakes", ["vision_native"]), ("raise_vs_fakes", ["raise"]),
                       ("both_pools_vs_fakes", ["vision_native", "raise"])]:
        sub = df[df.pool.isin(pools + ["fake"])].reset_index(drop=True)
        fb[tag] = format_bias_probe.format_bias_report(sub, side, SEED, n_boot=1000)
        fb[tag]["n_train_real"] = int(((sub.split == "train") & (sub.label == 0)).sum())
        fb[tag]["n_test_real"] = int(((sub.split == "test") & (sub.label == 0)).sum())
    b0r = json.loads((CACHE / "b0_results.json").read_text())
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    snap = list((ROOT / "data/hf_cache/hub/models--facebook--dinov2-base/snapshots").iterdir())[0]
    h = hashlib.sha256()
    with open(snap / "model.safetensors", "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    prep = json.loads((CACHE / "prep_report.json").read_text())
    results = dict(
        git=dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                 dirty_note="working-tree changes are this experiment's new files"),
        config=CFG, prep=prep,
        n_test=test.groupby(["pool", "generator"]).size().rename("n").reset_index().to_dict("records"),
        n_calib=len(calib), n_calib_by_pool=calib.pool.value_counts().to_dict(),
        dinov2=dict(repo="facebook/dinov2-base", revision=snap.name, license="Apache-2.0",
                    model_safetensors_sha256=h.hexdigest()),
        d008_guard=json.loads((CACHE / "d008_guard.json").read_text()),
        timings_seconds=json.loads((CACHE / "timings.json").read_text()),
        format_bias=fb, scorers=res,
        b0=dict(chosen_C=b0r["chosen_C"], n_train=b0r["n_train"], n_train_real=b0r["n_train_real"], logo=b0r["logo"]))
    jdump(results, HERE / "results.json")
    jdump(CFG, HERE / "config.json")
    write_tables(json.loads((HERE / "results.json").read_text()))
    log("report written")


def _f(m, k="auroc", ci=True, d=3):
    v = m[k]
    if v["value"] is None:
        return "n/a"
    s = f"{v['value']:.{d}f}"
    if ci and v["ci_lo"] is not None:
        s += f" [{v['ci_lo']:.{d}f}, {v['ci_hi']:.{d}f}]"
    return s


def write_tables(R):
    S = R["scorers"]
    names = list(S)
    L = []
    L.append("# 003-baselines-eval tables\n")
    L.append(f"git HEAD {R['git']['head']} · seed 0 · AUROC real pool = VISION-native test + RAISE test; fakes = 7 non-FLUX generators + FLUX (all test-split). "
             "B0 = DINOv2-base probe (FLUX never in train). 95% stratified bootstrap CIs (1000). Calibrated = threshold at 5% FPR on the calib reals of the SAME condition.\n")
    L.append("## Headline: AUROC and TPR@1%FPR per condition; FLUX (held-out) AUROC\n")
    L.append("| scorer | AUROC clean | AUROC jpeg75_resize05 | AUROC social_sim | TPR@1%FPR clean | TPR@1%FPR jpeg75_resize05 | TPR@1%FPR social_sim | FLUX AUROC clean / jpeg75_resize05 / social_sim |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        a = [f"{S[n][c]['overall']['auroc']['value']:.3f}" for c in CONDITIONS]
        t = [f"{S[n][c]['overall']['tpr_at_1fpr']['value']:.3f}" for c in CONDITIONS]
        fl = " / ".join(f"{S[n][c]['per_generator']['flux']['auroc']['value']:.3f}" for c in CONDITIONS)
        L.append(f"| {n} | " + " | ".join(a) + " | " + " | ".join(t) + f" | {fl} |")
    L.append("\n## Main table: scorer x condition (all fakes incl. FLUX vs VISION-native + RAISE test reals)\n")
    L.append("| scorer | condition | AUROC [95% CI] | AP | bal.acc @logit0 | bal.acc @5%FPR-calib | FPR reached @calib | FPR@95%TPR | TPR@1%FPR [CI] | AUROC excl. FLUX |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        for c in CONDITIONS:
            o = S[n][c]["overall"]
            L.append(f"| {n} | {c} | {_f(o)} | {_f(o,'ap',ci=False)} | {_f(o,'bal_acc',ci=False)} | {_f(o,'bal_acc_fpr5',ci=False)} | "
                     f"{_f(o,'fpr_at_fpr5_calib',ci=False)} | {_f(o,'fpr_at_95tpr',ci=False)} | {_f(o,'tpr_at_1fpr')} | "
                     f"{_f(S[n][c]['overall_noflux'])} |")
    L.append("\n## AUROC by real pool (all fakes vs that pool only)\n")
    L.append("| scorer | condition | VISION-native vs fakes | RAISE vs fakes |")
    L.append("|---|---|---|---|")
    for n in names:
        for c in CONDITIONS:
            p = S[n][c]["per_real_pool"]
            L.append(f"| {n} | {c} | {_f(p['vision_native'])} | {_f(p['raise'])} |")
    L.append("\n## Per-generator AUROC (each generator vs the same real pool: VISION-native + RAISE test). FLUX = held out for B0\n")
    L.append("| scorer | condition | " + " | ".join(FAKE_GENS) + " |")
    L.append("|---|---|" + "---|" * len(FAKE_GENS))
    for n in names:
        for c in CONDITIONS:
            pg = S[n][c]["per_generator"]
            L.append(f"| {n} | {c} | " + " | ".join(f"{pg[g]['auroc']['value']:.3f}" for g in FAKE_GENS) + " |")
    L.append("\n## Per-real-source FPR at the calibrated threshold (5% FPR on calib reals of the same condition); k/n shown. WA/FB-HQ reals are platform-shared copies of the test natives\n")
    L.append("| scorer | condition | thr | VISION-native | RAISE | VISION-WhatsApp | VISION-Facebook-HQ |")
    L.append("|---|---|---|---|---|---|---|")
    for n in names:
        for c in CONDITIONS:
            fs = S[n][c]["fpr_by_source"]
            cells = [f"{fs[p]['calib5']['fpr']:.3f} ({fs[p]['calib5']['k']}/{fs[p]['calib5']['n']})" for p in POOLS_REAL]
            L.append(f"| {n} | {c} | {S[n][c]['thresholds']['calib5']:.2f} | " + " | ".join(cells) + " |")
    L.append("\n### Same, at the native threshold (logit 0)\n")
    L.append("| scorer | condition | VISION-native | RAISE | VISION-WhatsApp | VISION-Facebook-HQ |")
    L.append("|---|---|---|---|---|---|")
    for n in names:
        for c in CONDITIONS:
            fs = S[n][c]["fpr_by_source"]
            L.append(f"| {n} | {c} | " + " | ".join(f"{fs[p]['native0']['fpr']:.3f}" for p in POOLS_REAL) + " |")
    L.append("\n### Same, at the threshold calibrated on CLEAN calib reals (deployment view: threshold not re-fit per condition); TPR over all fakes\n")
    L.append("| scorer | condition | thr | VISION-native | RAISE | VISION-WhatsApp | VISION-Facebook-HQ | TPR all fakes |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        for c in CONDITIONS:
            fs = S[n][c]["fpr_by_source"]
            L.append(f"| {n} | {c} | {S[n][c]['thresholds']['clean_calib5']:.2f} | " +
                     " | ".join(f"{fs[p]['clean_calib5']['fpr']:.3f}" for p in POOLS_REAL) +
                     f" | {S[n][c]['tpr_at_clean_calib5']:.3f} |")
    L.append("\n## B0 (DINOv2-base CLS+mean probe) leave-one-generator-out AUROC (probe trained on clean; test g = all its fakes vs VISION-native+RAISE test reals; FLUX is never trained on and is not a LOGO fold)\n")
    lg = R["b0"]["logo"]
    L.append("| held-out generator | " + " | ".join(CONDITIONS) + " |")
    L.append("|---|" + "---|" * len(CONDITIONS))
    for g in [g for g in FAKE_GENS if g != "flux"]:
        L.append(f"| {g} | " + " | ".join(_f(lg[c]["folds"][g]) for c in CONDITIONS) + " |")
    L.append("| macro-mean AUROC | " + " | ".join(f"{lg[c]['macro']['auroc']:.3f}" for c in CONDITIONS) + " |")
    L.append("| macro TPR@1%FPR | " + " | ".join(f"{lg[c]['macro']['tpr_at_1fpr']:.3f}" for c in CONDITIONS) + " |")
    L.append(f"\n(B0 standard probe: n_train={R['b0']['n_train']} of which {R['b0']['n_train_real']} reals, C={R['b0']['chosen_C']}; LOGO folds choose their own C.)\n")
    L.append("## Format-bias probe (bytes/pixel, quant-table presence + quality, dims of ORIGINAL files, HF energy; logistic regression train split -> test split; fakes = all non-LDM generators incl. FLUX in test)\n")
    L.append("| real pool | AUROC [95% CI] | n train real | n test real | leakage flag (CI lower > 0.6) |")
    L.append("|---|---|---|---|---|")
    for k, v in R["format_bias"].items():
        L.append(f"| {k} | {v['auroc']:.3f} [{v['auroc_ci'][0]:.3f}, {v['auroc_ci'][1]:.3f}] | {v['n_train_real']} | {v['n_test_real']} | {v['leakage_flag']} |")
    L.append("\nStandardized coefficients (" + ", ".join(["bytes_per_pixel", "has_quant_table", "jpeg_quality", "width", "height", "hf_energy"]) + "):")
    for k, v in R["format_bias"].items():
        L.append(f"- {k}: {v['standardized_coef']}")
    L.append("\n## D-008 guard (max |dlogit| MPS vs CPU on fixed 32-image subset; tol 1e-3; B0 = decision logit of the probe)\n")
    L.append("| run | max abs dlogit | ok |")
    L.append("|---|---|---|")
    for k, v in R["d008_guard"].items():
        L.append(f"| {k} | {v['max_abs_dlogit']:.2e} | {v['ok']} |")
    (HERE / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:] or ["all"]
    cmd = a[0]
    if cmd == "prep":
        stage_prep(force="--force" in a)
    elif cmd == "cpuref":
        stage_cpuref(a[1])
    elif cmd == "score":
        stage_score([x for x in a[1:] if not x.startswith("-")] or None)
    elif cmd == "b0":
        stage_b0()
    elif cmd == "report":
        stage_report()
    elif cmd == "all":
        stage_prep()
        stage_score()
        stage_b0()
        stage_report()
    else:
        raise SystemExit(__doc__)
