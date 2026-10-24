"""E3 PILOT (docs/04-phase4-plan.md Amendment 9): does test-time canonicalization (jpeg75_resize05) undo RAID transfer
attacks? B0 (A0 recipe, DINOv2-base) + CommFor / RINE / ClipDet references. See notes.md for design and caveats.

Stages (resumable; all caches in cache/, smoke runs only in cache/smoke/):
  verify    check the 1,500 files of raid_manifest.csv exist and match their sha256 (manifest is never modified)
  b0fit     fit the B0 probe (A0 recipe, all pilot b0_train rows) and check it reproduces 005 pilot_v2/orig A0/full
  score     score every detector on the 6 input sets (+ D-008 guard on CUDA runs)
  cpuref    (internal) CPU reference scores of the fixed 32-item guard subset for one detector
  export    metrics, bootstrap, B0 label, tables -> pilot/ (or cache/smoke/out/)
  selftest  synthetic checks of the metric/label code (no models, no data)
  all       verify, b0fit, score, export

  CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/010-raid-transfer/run.py --stage all --device cpu --smoke 5 --detectors b0
  env/.venv/bin/python -I experiments/010-raid-transfer/run.py --stage all --device cuda        # full run
"""
import argparse
import hashlib
import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "002-baselines"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from PIL import Image

import metrics

MANIFEST = HERE / "raid_manifest.csv"
SPLITS = ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv"
CFG004 = ROOT / "experiments/004-phase4-features/config.json"
REF005 = ROOT / "experiments/005-nuisance-projection/pilot_v2/orig/cache/scores_A0.npz"
SEED = 0
VER = None            # preprocessing version (cache key), set in configure()
BATCH = 16            # modest on purpose: the GPU is shared with other jobs
TOL = 1e-3
N_BOOT = 2000
SIZE = 256
CLASSES = ["real", "gen_0", "gen_1", "gen_2", "gen_3"]
GENS = CLASSES[1:]
SETS = ["clean", "clean_canon", "eps8", "eps8_canon", "eps16", "eps16_canon"]
DETECTORS = ["b0", "communityforensics", "rine", "clipdet"]
SCORERS = {"communityforensics": ("scorers.communityforensics", "score"),
           "rine": ("scorers.rine", "score"),
           "clipdet": ("scorers.clipdet", "score")}
RAID_T4 = "RAID Table 4 (arXiv 2506.03988, via kb/datasets/raid.md): DINOv2 accuracy 83.1 -> 54.6 at eps 16/255"

# set later by configure()
CACHE = HERE / "cache"
OUT = HERE / "pilot"
LOGFILE = None
DEVICE = "cpu"
SMOKE = 0


def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    if LOGFILE is not None:
        with open(LOGFILE, "a") as f:
            f.write(line + "\n")


def configure(args):
    global CACHE, OUT, LOGFILE, DEVICE, SMOKE, VER
    VER = prep_version()
    DEVICE, SMOKE = args.device, args.smoke
    CACHE = HERE / "cache" / "smoke" if SMOKE else HERE / "cache"
    OUT = CACHE / "out" if SMOKE else HERE / "pilot"
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    gi = HERE / "cache" / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n")
    LOGFILE = OUT / "run.log"
    os.environ["DD_DEVICE"] = DEVICE            # read by scorers._common.pick_device
    import torch
    torch.set_num_threads(args.threads)
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), "--device cuda but CUDA is not available"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        assert DEVICE == "cpu"


# ----------------------------------------------------------------------------- data
def load_manifest():
    m = pd.read_csv(MANIFEST, dtype={"id": str}, keep_default_na=False)
    assert len(m) == 1500 and set(m.cls) == set(CLASSES) and set(m.kind) == {"original", "eps8", "eps16"}
    return m


def build_items(m):
    """One row per (cls, id): paths of original / eps8 / eps16. Ordered by class, then id; the same sorted id list
    in every class (asserted). Smoke: first N ids per class."""
    ids = sorted(set(m.id[m.cls == "real"]))
    assert len(ids) == 100
    for c in CLASSES:
        assert sorted(set(m.id[m.cls == c])) == ids, f"id sets differ for {c}"
        assert (m[m.cls == c].groupby("id").kind.nunique() == 3).all()
    p = m.pivot(index=["cls", "id"], columns="kind", values="path")
    size = {}
    for c in CLASSES:
        for i in sorted(set(m.id)):
            with Image.open(ROOT / p.loc[(c, i), "original"]) as im:
                size[(c, i)] = im.size
    if SMOKE:   # N ids with at least ceil(N/2) unpadded reals so that the primary analysis exists
        un = [i for i in ids if min(size[("real", i)]) >= SIZE]
        pad = [i for i in ids if min(size[("real", i)]) < SIZE]
        k = (SMOKE + 1) // 2
        ids = sorted(un[:k] + pad[:SMOKE - k]) if len(un) >= k else ids[:SMOKE]
    rows = [dict(key=f"{c}/{i}", cls=c, id=i, label=int(c != "real"), generator="" if c == "real" else c,
                 original=p.loc[(c, i), "original"], eps8=p.loc[(c, i), "eps8"], eps16=p.loc[(c, i), "eps16"],
                 orig_w=size[(c, i)][0], orig_h=size[(c, i)][1], padded=bool(min(size[(c, i)]) < SIZE))
            for c in CLASSES for i in ids]
    df = pd.DataFrame(rows)
    assert df.key.is_unique and len(df) == 5 * len(ids)
    return df


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def center_crop_256(img):
    """RAID's own pre-attack preprocessing (verified by the pairing check): plain center crop to 256x256, no resize;
    zero-padded when a side is under 256 (torchvision CenterCrop semantics)."""
    from torchvision import transforms as T
    return T.CenterCrop(SIZE)(img.convert("RGB"))


def canon(img):
    """jpeg75_resize05 (lib/perturb) then bicubic back up to 256 so every detector sees its nominal field (D1)."""
    from perturb import jpeg75_resize05
    return jpeg75_resize05(img).resize((SIZE, SIZE), Image.BICUBIC)


def make_input(row, setname):
    base, _, c = setname.partition("_")
    if base == "clean":
        with Image.open(ROOT / row["original"]) as im:
            im.load()
            img = center_crop_256(im)
    else:
        with Image.open(ROOT / row[base]) as im:
            im.load()
            img = im.convert("RGB")
        assert img.size == (SIZE, SIZE), (row["key"], img.size)
    return canon(img) if c == "canon" else img


def prep_version():
    """Cache key part: changes whenever the preprocessing code or SIZE changes."""
    src = "".join(inspect.getsource(f) for f in (center_crop_256, canon, make_input)) + f"|{SIZE}|jpeg75_resize05"
    return "p" + hashlib.sha1(src.encode()).hexdigest()[:8]


def pairing_check(items):
    """RAID attacks the plain 256 center crop: for EVERY id the mean |attacked - clean| (grey levels) must be <= 10
    at eps 8 and <= 18 at eps 16, else abort. Returns the statistics."""
    lim = dict(eps8=10.0, eps16=18.0)
    rows = []
    for r in items.to_dict("records"):
        c = np.asarray(make_input(r, "clean"), np.int16)
        d = {e: np.abs(np.asarray(make_input(r, e), np.int16) - c) for e in lim}
        rows.append(dict(key=r["key"], cls=r["cls"], mean8=d["eps8"].mean(), max8=int(d["eps8"].max()),
                         mean16=d["eps16"].mean(), max16=int(d["eps16"].max())))
    t = pd.DataFrame(rows)
    stats = dict(n_items=len(t), limits_mean_abs_grey=lim,
                 eps8=dict(max_item_mean=float(t.mean8.max()), median_item_mean=float(t.mean8.median()), max_pixel=int(t.max8.max())),
                 eps16=dict(max_item_mean=float(t.mean16.max()), median_item_mean=float(t.mean16.median()), max_pixel=int(t.max16.max())),
                 per_class_max_item_mean={c: dict(eps8=float(g.mean8.max()), eps16=float(g.mean16.max())) for c, g in t.groupby("cls")})
    bad = t[(t.mean8 > lim["eps8"]) | (t.mean16 > lim["eps16"])]
    if len(bad):
        raise SystemExit(f"PAIRING CHECK FAILED for {len(bad)} items (clean control != attacked image's source), e.g. {bad.key.tolist()[:3]}")
    return stats


def stage_verify(items):
    m = load_manifest()
    bad = [r.path for r in m.itertuples() if not (ROOT / r.path).is_file() or sha256_file(ROOT / r.path) != r.sha256]
    assert not bad, f"{len(bad)} manifest files missing or with wrong sha256, e.g. {bad[:3]}"
    assert (m.sha256 == m.hf_lfs_sha256).all()
    log(f"verify: {len(m)} files exist and match the manifest sha256 (== HF LFS oid)")
    st = pairing_check(items)
    log(f"pairing check OK ({st['n_items']} items): eps8 max item-mean {st['eps8']['max_item_mean']:.2f} (limit 10), "
        f"eps16 max item-mean {st['eps16']['max_item_mean']:.2f} (limit 18); max pixel diff {st['eps8']['max_pixel']}/{st['eps16']['max_pixel']}")
    (CACHE / "verify_ok.json").write_text(json.dumps(dict(n=len(m), ok=True, pairing=st, prep_version=VER), indent=1))


# ----------------------------------------------------------------------------- B0
def load_b0_train():
    df = pd.read_csv(SPLITS, keep_default_na=False, dtype={"vidx": str}).astype({"in_scoring_set": bool, "b0_train": bool})
    from p4features import load_features_aligned
    X = load_features_aligned(df, "primary", "clean", subdir="pilot")
    tr = df.b0_train.to_numpy()
    assert tr.sum() > 100 and not np.isnan(X[tr]).any()
    return df, X, tr


def stage_b0fit():
    p = CACHE / "b0_probe.pkl"
    if p.exists():
        log("b0fit: probe cached")
        return
    import probe
    df, X, tr = load_b0_train()
    y = df.label.to_numpy()
    mdl = probe.fit_probe(X[tr], y[tr], SEED, class_weight="balanced", groups=df.sha256.to_numpy()[tr])
    info = dict(n_train=int(tr.sum()), n_real=int((y[tr] == 0).sum()), n_fake=int((y[tr] == 1).sum()),
                generators=sorted(set(df.generator[tr & (y == 1)])), C=float(mdl[-1].C_[0]))
    if not REF005.exists() and not SMOKE:
        raise SystemExit(f"missing {REF005}: cannot check the B0 refit")
    if REF005.exists():       # reproduce 005 pilot_v2/orig A0/full on the clean condition (index 0)
        ref = np.load(REF005)["full"][0]
        ok = ~np.isnan(X).any(1)
        d = float(np.abs(mdl.decision_function(X[ok]) - ref[ok]).max())
        info["max_abs_diff_vs_005_A0_full_clean"] = d
        log(f"b0fit: max|score - 005 A0/full clean| = {d:.2e} over {int(ok.sum())} rows")
        if d >= 1e-3:
            raise SystemExit(f"B0 refit differs from 005 A0/full by {d:.2e} >= 1e-3")
    with open(p, "wb") as f:
        pickle.dump(mdl, f)
    (CACHE / "b0_probe_info.json").write_text(json.dumps(info, indent=1))
    log("b0fit:", info)


class B0Scorer:
    """DINOv2-base (004 config: primary = last block CLS+mean, 224 center crop, no resize) -> A0 probe logit."""

    def __init__(self):
        from features import load_backbone
        c = json.loads(CFG004.read_text())
        pr = c["primary"]
        assert c["spec"] == "hf:dinov2:facebook/dinov2-base" and pr["layers"] == [-1] and pr["use_cls"] and pr["use_mean"]
        self.bb = load_backbone(c["spec"], input_size=pr["input_size"], layers=tuple(pr["layers"]), use_cls=pr["use_cls"],
                                use_mean=pr["use_mean"], crop_mode=pr["crop_mode"])
        with open(CACHE / "b0_probe.pkl", "rb") as f:
            self.probe = pickle.load(f)

    def __call__(self, imgs):
        from features import embed_images
        F = embed_images(self.bb, imgs, BATCH, DEVICE)
        if getattr(self.bb, "force_cpu", False) and DEVICE != "cpu":
            raise SystemExit("accelerator forward fell back to CPU; aborting (D-010)")
        return self.probe.decision_function(F.astype(np.float64))


def get_fn(name):
    if name == "b0":
        return B0Scorer()
    m, f = SCORERS[name]
    return getattr(importlib.import_module(m), f)


# ----------------------------------------------------------------------------- scoring + D-008 guard
def check_subset(items):
    p = CACHE / "check_subset.json"
    if p.exists():
        return json.loads(p.read_text())
    r, f = sorted(items.key[items.label == 0]), sorted(items.key[items.label == 1])
    k = min(16, len(r), len(f))
    rng = np.random.default_rng(SEED)
    sub = [r[i] for i in rng.choice(len(r), k, replace=False)] + [f[i] for i in rng.choice(len(f), k, replace=False)]
    p.write_text(json.dumps(sub))
    return sub


def score_rows(fn, rows, setname, chunk=BATCH):
    out = {}
    for i in range(0, len(rows), chunk):
        ch = rows.iloc[i:i + chunk]
        imgs = [make_input(r, setname) for r in ch.to_dict("records")]
        sc = np.asarray(fn(imgs), float).reshape(-1)
        assert len(sc) == len(imgs) and np.isfinite(sc).all()
        out.update(zip(ch.key, sc))
    return out


def score_path(name, setname):
    """Score caches are keyed by (detector, input set, preprocessing version, device)."""
    return CACHE / f"scores_{name}_{setname}_{VER}_{DEVICE}.npz"


def load_scores(name, setname):
    p = score_path(name, setname)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        assert str(z["ver"]) == VER and str(z["device"]) == DEVICE, f"cache key mismatch in {p}"
        return dict(zip(z["keys"].tolist(), z["vals"].tolist()))
    return {}


def save_scores(name, setname, d):
    p = score_path(name, setname)
    ks = sorted(d)
    np.savez(str(p) + ".tmp.npz", keys=np.array(ks), vals=np.array([d[k] for k in ks]), ver=VER, device=DEVICE)
    os.replace(str(p) + ".tmp.npz", p)


def cpuref_path(name):
    return CACHE / f"cpuref_{name}_{VER}.json"


def stage_cpuref(name, items):
    assert DEVICE == "cpu"
    sub = check_subset(items)
    ii = items.set_index("key", drop=False)
    fn = get_fn(name)
    p = cpuref_path(name)
    ref = json.loads(p.read_text()) if p.exists() else {}
    for s in SETS:
        if s in ref:
            continue
        t = time.time()
        ref[s] = score_rows(fn, ii.loc[sub], s, chunk=4)
        p.write_text(json.dumps(ref))
        log(f"cpuref {name} {s} done in {time.time() - t:.0f}s")


def guard_done(guard, name, s, sc, items):
    g = guard.get(f"{name}/{s}")
    return bool(g and g.get("ver") == VER and g.get("device") == DEVICE and g.get("ok") and len(sc) == len(items))


def stage_score(names, items, args):
    ii = items.set_index("key", drop=False)
    sub = check_subset(items)
    rest = items[~items.key.isin(sub)]
    gp = CACHE / f"d008_guard_{VER}_{DEVICE}.json"
    guard = json.loads(gp.read_text()) if gp.exists() else {}
    for name in names:
        todo_sets = [s for s in SETS if not (DEVICE == "cuda" and guard_done(guard, name, s, load_scores(name, s), items))
                     and not (DEVICE == "cpu" and len(load_scores(name, s)) == len(items))]
        if not todo_sets:
            log(f"{name}: complete")
            continue
        if DEVICE == "cuda":
            # the D-008 reference is ALWAYS computed fresh on CPU (never reused from an earlier run)
            cpuref_path(name).unlink(missing_ok=True)
            log(f"{name}: fresh CPU reference ({len(sub)} items x {len(SETS)} sets)")
            cmd = [sys.executable, "-I", str(HERE / "run.py"), "--stage", "cpuref", "--detectors", name,
                   "--device", "cpu", "--threads", str(args.threads)] + (["--smoke", str(SMOKE)] if SMOKE else [])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            if subprocess.run(cmd, env=env).returncode != 0:
                raise SystemExit(f"cpuref failed for {name}")
            ref = json.loads(cpuref_path(name).read_text())
        fn = get_fn(name)
        for s in todo_sets:
            sc = load_scores(name, s)
            if DEVICE == "cuda":
                sc.update(score_rows(fn, ii.loc[sub], s))     # guard subset rescored fresh on the GPU, always
                mx = float(np.abs([sc[k] - ref[s][k] for k in sub]).max())
                guard[f"{name}/{s}"] = dict(max_abs_dlogit=mx, n=len(sub), tol=TOL, ok=bool(mx <= TOL), device="cuda", ver=VER)
                log(f"{name} {s}: D-008 max|dlogit|={mx:.2e} (tol {TOL})")
                gp.write_text(json.dumps(guard, indent=1))
                if mx > TOL:
                    raise SystemExit(f"D-008 ABORT: {name}/{s} max |dlogit| {mx:.3e} > {TOL}")
            else:
                todo_sub = ii.loc[[k for k in sub if k not in sc]]
                if len(todo_sub):
                    sc.update(score_rows(fn, todo_sub, s))
                guard[f"{name}/{s}"] = dict(applicable=False, device="cpu", ver=VER,
                                            note="scored on CPU: the D-008 reference is itself CPU, nothing to compare")
                gp.write_text(json.dumps(guard, indent=1))
            save_scores(name, s, sc)
            todo = rest[~rest.key.isin(sc)]
            for i in range(0, len(todo), 100):
                sc.update(score_rows(fn, todo.iloc[i:i + 100], s))
                save_scores(name, s, sc)
                log(f"{name} {s}: {len(sc)}/{len(items)}")


# ----------------------------------------------------------------------------- metrics
def b0_label(a_clean, drop, rec):
    """Amendment 9 table. rec = recovery (None if undefined)."""
    if a_clean < 0.80:
        return "UNINFORMATIVE"
    if drop > 0.10:
        return "FAIL" if (rec is None or rec < 0.5) else "INCONCLUSIVE"
    if drop < 0.02 or (rec is not None and rec >= 0.5):
        return "PASS"
    return "INCONCLUSIVE"


class Plan:
    """Rows entering one analysis and its bootstrap plan. Reals and each generator's fakes are resampled
    INDEPENDENTLY, by id within class (lib/metrics.bootstrap_weights: reals by cluster = id, fakes by image
    stratified per generator). The same plan is used for every input set and detector (paired)."""

    def __init__(self, items, real_mask, fake_mask, n_boot=N_BOOT, seed=SEED):
        cls = items.cls.to_numpy()
        self.R = np.where((cls == "real") & real_mask)[0]
        self.F = np.where((cls != "real") & fake_mask)[0]          # class-major: each generator's rows are contiguous
        self.gens = [g for g in GENS if (cls[self.F] == g).any()]
        self.cols = {g: np.where(cls[self.F] == g)[0] for g in self.gens}
        assert len(self.R) >= 2 and all(len(c) >= 2 for c in self.cols.values()), "too few rows for a bootstrap"
        self.Wr, self.Wf = metrics.bootstrap_weights(items.id.to_numpy()[self.R], cls[self.F], n_boot, seed)
        self.y = np.r_[np.zeros(len(self.R)), np.ones(len(self.F))].astype(int)
        self.counts = dict(n_real=len(self.R), n_fake=len(self.F), n_fake_per_generator={g: len(c) for g, c in self.cols.items()})


def auc_pack(S, pl):
    """S: scores over items. Point value + bootstrap replicates, pooled and per generator."""
    sr, sf = S[pl.R], S[pl.F]
    one_r = np.ones((1, len(sr)))
    out = {}
    for g in pl.gens:
        c = pl.cols[g]
        out[g] = (float(metrics.weighted_auroc(sr, sf[c], one_r, np.ones((1, len(c))))[0]),
                  metrics.weighted_auroc(sr, sf[c], pl.Wr, pl.Wf[:, c]))
    out["pooled"] = (float(metrics.weighted_auroc(sr, sf, one_r, np.ones((1, len(sf))))[0]),
                     metrics.weighted_auroc(sr, sf, pl.Wr, pl.Wf))
    return out


def ci(x, nan=False):
    f = np.nanpercentile if nan else np.percentile
    lo, hi = f(x, [2.5, 97.5])
    return float(lo), float(hi)


def derived(A, eps):
    """A[set] = pooled (point, boot). Returns drop/recovery/gain/clean cost with CIs."""
    ac, ae, aec, acc = A["clean"], A[eps], A[f"{eps}_canon"], A["clean_canon"]
    drop_p, gain_p = ac[0] - ae[0], aec[0] - ae[0]
    drop_b, gain_b = ac[1] - ae[1], aec[1] - ae[1]
    rec_p = gain_p / drop_p if drop_p > 1e-9 else None
    ok = drop_b > 0.02
    rec_b = np.where(ok, gain_b / np.where(ok, drop_b, 1), np.nan)
    cost_p, cost_b = ac[0] - acc[0], ac[1] - acc[1]
    d = dict(auroc_clean=ac[0], auroc_attacked=ae[0], auroc_attacked_canon=aec[0], auroc_clean_canon=acc[0],
             drop=dict(value=drop_p, ci=ci(drop_b)), gain=dict(value=gain_p, ci=ci(gain_b)),
             recovery=dict(value=rec_p, ci=ci(rec_b, True) if ok.sum() > 20 else None, n_valid_boot=int(ok.sum()),
                           note="bootstrap CI only over replicates with drop > 0.02"),
             canon_clean_cost=dict(value=cost_p, ci=ci(cost_b)))
    return d, drop_b, rec_b, ok


def analyze_detector(S, pl, label=False):
    """S[set] -> array over items. label=True adds the Amendment 9 B0 label (primary analysis of B0 only)."""
    P = {s: auc_pack(S[s], pl) for s in SETS}
    res = dict(counts=pl.counts,
               auroc={s: {k: dict(value=v[0], ci=ci(v[1])) for k, v in P[s].items()} for s in SETS})
    A = {s: P[s]["pooled"] for s in SETS}
    res["bal_acc_at_logit0"] = {s: float(metrics.balanced_accuracy(pl.y, np.r_[S[s][pl.R], S[s][pl.F]], 0.0)) for s in SETS}
    d8, drop_b, rec_b, ok = derived(A, "eps8")
    d16, _, _, _ = derived(A, "eps16")
    res["eps8"], res["eps16_descriptive"] = d8, d16
    if label:
        rec = d8["recovery"]["value"]
        lab = b0_label(d8["auroc_clean"], d8["drop"]["value"], rec)
        reps = [b0_label(A["clean"][1][b], drop_b[b], None if not ok[b] else float(rec_b[b])) for b in range(len(drop_b))]
        share = {k: float(np.mean([r == k for r in reps])) for k in ("UNINFORMATIVE", "PASS", "FAIL", "INCONCLUSIVE")}
        res["b0_label"] = dict(label=lab, rule="Amendment 9 + clarification 2 (primary = unpadded reals vs unpadded fakes)",
                               point=dict(auroc_clean=d8["auroc_clean"], drop=d8["drop"]["value"], recovery=rec),
                               bootstrap_label_share=share)
    return res


def make_plans(items):
    pad = items.padded.to_numpy()
    return dict(primary=Plan(items, ~pad, ~pad),                  # unpadded reals vs unpadded fakes (labelled)
                sensitivity_all=Plan(items, np.ones(len(items), bool), np.ones(len(items), bool)))   # never labelled


def load_all_scores(items, names):
    S = {}
    for n in names:
        S[n] = {}
        for s in SETS:
            sc = load_scores(n, s)
            miss = [k for k in items.key if k not in sc]
            assert not miss, f"incomplete scores {n}/{s} ({DEVICE}, {VER}): {len(miss)} items missing"
            S[n][s] = np.array([sc[k] for k in items.key])
    return S


def fmt(v, c=None, p=3):
    return f"{v:.{p}f}" + (f" [{c[0]:.{p}f}, {c[1]:.{p}f}]" if c else "")


def write_tables(R, names, plans, pairing):
    pc, sc = plans["primary"].counts, plans["sensitivity_all"].counts
    L = [f"# E3 PILOT: RAID transfer attacks vs canonicalization{' (SMOKE, not results)' if SMOKE else ''}", "",
         f"Device {DEVICE}, preprocessing {VER}. 95% CIs: bootstrap over ids ({N_BOOT} resamples, seed {SEED}), reals and each generator's fakes resampled independently.",
         "Pooled AUROC = real vs the generators pooled. Higher score = more fake. Pilot: does not count toward the stop rule.", "",
         "## Counts", "",
         f"- PRIMARY (labelled): unpadded reals vs unpadded fakes (both sides >= 256 px in the original): {pc['n_real']} reals, {pc['n_fake']} fakes {pc['n_fake_per_generator']}.",
         f"- SENSITIVITY (never labelled): all {sc['n_real']} reals ({sc['n_real'] - pc['n_real']} padded, with black borders after RAID's zero-pad crop) vs {sc['n_fake']} fakes.", ""]
    if pairing:
        L += [f"- Pairing check (clean control vs attacked, mean |diff| in grey levels per item): eps8 max {pairing['eps8']['max_item_mean']:.2f} "
              f"(limit 10), eps16 max {pairing['eps16']['max_item_mean']:.2f} (limit 18); max pixel diff {pairing['eps8']['max_pixel']} / {pairing['eps16']['max_pixel']}.", ""]
    if "b0" in R:
        b = R["b0"]["primary"]["b0_label"]
        p = b["point"]
        L += ["## B0 label (Amendment 9, primary set)", "", f"**{b['label']}**  (AUROC clean {p['auroc_clean']:.3f}, drop {p['drop']:.3f}, "
              f"recovery {'n/a' if p['recovery'] is None else format(p['recovery'], '.3f')}); bootstrap label shares: "
              + ", ".join(f"{k} {v:.2f}" for k, v in b["bootstrap_label_share"].items()), ""]
    for key, title in (("primary", "PRIMARY (unpadded)"), ("sensitivity_all", "SENSITIVITY (all 100 reals, unlabelled)")):
        L += [f"## {title}: pooled AUROC by input set", "", "| detector | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
        for n in names:
            a = R[n][key]["auroc"]
            L.append(f"| {n} | " + " | ".join(fmt(a[s]["pooled"]["value"], a[s]["pooled"]["ci"]) for s in SETS) + " |")
        for dk, dt in (("eps8", "eps 8/255"), ("eps16_descriptive", "eps 16/255 (descriptive)")):
            L += ["", f"### {title}: drop / recovery / canonicalization cost, {dt}", "",
                  "| detector | drop | gain (canon on attacked) | recovery | canon clean cost |", "|---|---|---|---|---|"]
            for n in names:
                d = R[n][key][dk]
                rec = d["recovery"]
                rs = "n/a" if rec["value"] is None else fmt(rec["value"], rec["ci"]) + f" (n={rec['n_valid_boot']})"
                L.append(f"| {n} | {fmt(d['drop']['value'], d['drop']['ci'])} | {fmt(d['gain']['value'], d['gain']['ci'])} | {rs} "
                         f"| {fmt(d['canon_clean_cost']['value'], d['canon_clean_cost']['ci'])} |")
        L.append("")
    L += ["## PRIMARY per-generator AUROC (point values; CIs in results.json)", "",
          "| detector | set | " + " | ".join(GENS) + " |", "|---|---|" + "---|" * len(GENS)]
    for n in names:
        for s in SETS:
            a = R[n]["primary"]["auroc"][s]
            L.append(f"| {n} | {s} | " + " | ".join(f"{a[g]['value']:.3f}" if g in a else "n/a" for g in GENS) + " |")
    L += ["", "## PRIMARY balanced accuracy at logit 0 (descriptive)", "", "| detector | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{R[n]['primary']['bal_acc_at_logit0'][s]:.3f}" for s in SETS) + " |")
    L += ["", f"Anchor: {RAID_T4}. Our probe, preprocessing and balanced-accuracy-at-0 differ; compare only qualitatively.",
          "", "Caveats: see notes.md (D1 canon upsample, clean control = RAID center crop, originals already JPEG ~q75, RAID generator mapping unverified)."]
    (OUT / "tables.md").write_text("\n".join(L) + "\n")


def stage_export(names, items):
    n_ids = int((items.cls == "real").sum())
    assert (items.cls.to_numpy() == np.repeat(CLASSES, n_ids)).all()
    S = load_all_scores(items, names)
    plans = make_plans(items)
    R = {n: {k: analyze_detector(S[n], pl, label=(n == "b0" and k == "primary")) for k, pl in plans.items()} for n in names}
    gp = CACHE / f"d008_guard_{VER}_{DEVICE}.json"
    guard = json.loads(gp.read_text()) if gp.exists() else {}
    (OUT / "d008_guard.json").write_text(json.dumps(guard, indent=1))
    long = pd.concat([pd.DataFrame(dict(detector=n, input_set=s, cls=items.cls.values, id=items.id.values,
                                        label=items.label.values, padded=items.padded.values, score=S[n][s]))
                      for n in names for s in SETS], ignore_index=True)
    long.to_csv(OUT / "scores_long.csv.gz", index=False)
    cfg = json.loads((HERE / "config.json").read_text())
    vp = CACHE / "verify_ok.json"
    pairing = json.loads(vp.read_text()).get("pairing") if vp.exists() else None
    res = dict(PILOT=True, smoke=SMOKE, device=DEVICE, prep_version=VER, n_ids_per_class=n_ids, detectors=names, input_sets=SETS,
               bootstrap=cfg["bootstrap"], counts={k: pl.counts for k, pl in plans.items()}, pairing_check=pairing,
               d008_guard=guard, results=R, raid_table4_anchor=RAID_T4,
               b0_probe=json.loads((CACHE / "b0_probe_info.json").read_text()) if (CACHE / "b0_probe_info.json").exists() else None,
               git_head=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_tables(R, names, plans, pairing)
    log(f"export: wrote results.json, tables.md, scores_long.csv.gz, d008_guard.json to {OUT}")


# ----------------------------------------------------------------------------- selftest
def stage_selftest():
    # label table (Amendment 9)
    assert b0_label(0.79, 0.3, 0.0) == "UNINFORMATIVE"
    assert b0_label(0.9, 0.15, 0.2) == "FAIL" and b0_label(0.9, 0.15, 0.6) == "INCONCLUSIVE"
    assert b0_label(0.9, 0.10, 0.5) == "PASS" and b0_label(0.9, 0.10, 0.49) == "INCONCLUSIVE"
    assert b0_label(0.9, 0.01, None) == "PASS" and b0_label(0.9, -0.05, None) == "PASS"
    assert b0_label(0.9, 0.05, 0.1) == "INCONCLUSIVE"
    # plan + metrics on synthetic scores: perfect clean, attack inverts half, canon restores
    rng = np.random.default_rng(0)
    n = 40
    items = pd.DataFrame(dict(cls=np.repeat(CLASSES, n), id=[f"{i:04d}" for _ in CLASSES for i in range(n)],
                              label=np.r_[np.zeros(n), np.ones(4 * n)].astype(int)))
    items["padded"] = (items.cls == "real") & (items.id.astype(int) < 15)       # 15 padded reals
    pl = make_plans(items)
    assert pl["primary"].counts["n_real"] == 25 and pl["primary"].counts["n_fake"] == 160
    assert pl["sensitivity_all"].counts["n_real"] == 40 and pl["sensitivity_all"].Wr.shape == (N_BOOT, 40)
    assert (pl["primary"].Wr.sum(1) == 25).all() and (pl["primary"].Wf[:, pl["primary"].cols["gen_1"]].sum(1) == n).all()
    assert not np.array_equal(pl["primary"].Wr[:, :40 - 15][:, :1], pl["primary"].Wf[:, :1])    # independent draws

    def mk(sep):
        s = rng.normal(size=5 * n)
        s[n:] += sep
        return s
    S = dict(clean=mk(3.0), clean_canon=mk(3.0), eps8=mk(0.5), eps8_canon=mk(2.5), eps16=mk(0.0), eps16_canon=mk(1.0))
    r = analyze_detector(S, pl["primary"], label=True)
    e = r["eps8"]
    assert e["auroc_clean"] > 0.95 and e["drop"]["value"] > 0.2 and 0.5 < e["recovery"]["value"] <= 1.2
    assert r["b0_label"]["label"] in ("PASS", "INCONCLUSIVE") and abs(sum(r["b0_label"]["bootstrap_label_share"].values()) - 1) < 1e-9
    assert "b0_label" not in analyze_detector(S, pl["sensitivity_all"])
    # point AUROC equals the rank formula
    from scipy.stats import rankdata
    pa = pl["primary"]
    sr, sf = S["eps8"][pa.R], S["eps8"][pa.F]
    rk = rankdata(np.r_[sr, sf])
    ref = (rk[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf))
    assert abs(auc_pack(S["eps8"], pa)["pooled"][0] - ref) < 1e-12
    # torchvision-style center crop: pads small images with zeros, never resizes
    im = center_crop_256(Image.new("RGB", (100, 300), (255, 255, 255)))
    a = np.asarray(im)
    assert im.size == (256, 256) and a[:, :78].max() == 0 and a[:, 78:178].min() == 255 and a[:, 178:].max() == 0
    im = center_crop_256(Image.new("RGB", (500, 400), (7, 7, 7)))
    assert im.size == (256, 256) and (np.asarray(im) == 7).all()
    assert prep_version().startswith("p")
    log("selftest OK")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["verify", "b0fit", "score", "cpuref", "export", "selftest", "all"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--smoke", type=int, default=0, metavar="N", help="N ids per class (<=10); outputs only in cache/smoke/")
    ap.add_argument("--detectors", nargs="+", default=DETECTORS, choices=DETECTORS)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.smoke and not 1 <= a.smoke <= 10:
        ap.error("--smoke N must be in 1..10")
    if a.stage == "selftest":
        stage_selftest()
        return
    configure(a)
    log(f"stage={a.stage} device={DEVICE} smoke={SMOKE} detectors={a.detectors} threads={a.threads}")
    items = build_items(load_manifest())
    st = a.stage
    if st in ("verify", "all"):
        stage_verify(items)
    if st in ("b0fit", "all"):
        stage_b0fit()
    if st == "cpuref":
        for n in a.detectors:
            stage_cpuref(n, items)
    if st in ("score", "all"):
        if "b0" in a.detectors:
            stage_b0fit()
        stage_score(a.detectors, items, a)
    if st in ("export", "all"):
        stage_export(a.detectors, items)


if __name__ == "__main__":
    main()
