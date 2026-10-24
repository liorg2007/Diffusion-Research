"""Critic check Q1 (control arm, exploratory, NOT pre-registered): `du2m` = du2 with the down/up factor fixed at k=2 for
the test natives (fakes' median k is 2.0; natives' 4.78), i.e. same field of view as orig and a fake-matched bandwidth.
Fakes keep their du2 scores. Natives are rebuilt in memory from the RAW file (no PNG written), conditioned with
perturb.apply_condition (same seed/key as load_cond), scored on CPU by CommFor/RINE/ClipDet and by the fixed B0 probes
(DINOv2 primary features on CPU, standardisation as run_du2.stage_b0).
Self-checks: (i) the in-memory pipeline with the per-image k reproduces the stored du2 PNG bit-exactly (8 natives);
(ii) CPU scores of those 8 du2 natives vs the cached CUDA du2 scores (D-008-style, tol 1e-3);
(iii) CPU B0 features of those 8 vs the 004 du2 feature cache.
Output: c4_scores.csv (sha, det, model, cond, arm=du2m|du2chk, score), c4_du2m.out (AUROC table with paired cluster CIs)."""
import os, sys, json, pickle, time
os.environ["DD_DEVICE"] = "cpu"
from pathlib import Path
HERE = Path(__file__).resolve().parent
E1 = HERE.parents[1]
sys.path.insert(0, str(E1))
import run as R                      # E1 module (chdir ROOT, sys.path, sp.CACHE); no side effects on import
R.SPL["du2"] = R.ROOT / "experiments/004-phase4-features/pilot/splits_pilot_du2.csv"
import numpy as np, pandas as pd
from PIL import Image, ImageOps
import torch
import preprocess as PP, perturb, metrics, p4features as P
ROOT = R.ROOT
OUTCSV = HERE / "c4_scores.csv"

def build(src, k_fixed=None):
    with Image.open(ROOT / src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w0, h0 = im.size
    if k_fixed is None:
        s = PP.fov_resize(im, 512)
    else:
        s = im.resize((round(w0 / k_fixed), round(h0 / k_fixed)), Image.BICUBIC)
    if s.size != (w0, h0):
        s = s.resize((w0, h0), Image.BICUBIC)
    return PP.center_crop(s, 512)

def main():
    df = R.load_df("orig")
    dfd = R.load_df("du2")
    nat = np.where((df.split == "test") & (df.pool == "vision_native"))[0]
    rows = df.iloc[nat]
    chk = rows.groupby("device").head(1).iloc[:8]
    # (i) pipeline reproduces stored du2
    for i, r in chk.iterrows():
        a = np.asarray(build(r.src_path)); b = np.asarray(Image.open(ROOT / dfd.path[i]))
        assert np.array_equal(a, b), f"in-memory du2 != stored du2 for {r.sha256}"
    log = open(HERE / "c4_du2m.log", "a")
    def L(*a):
        s = time.strftime("%H:%M:%S") + " " + " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()
    L("self-check (i): in-memory du2 == stored du2 PNG for", len(chk), "natives")
    have = pd.read_csv(OUTCSV) if OUTCSV.exists() else pd.DataFrame(columns=["sha256", "det", "model", "cond", "arm", "score"])
    done = set(zip(have.det, have.cond, have.arm))
    out = [have]
    imgs_m = {r.sha256: build(r.src_path, 2.0) for _, r in rows.iterrows()}
    imgs_c = {r.sha256: build(r.src_path) for _, r in chk.iterrows()}
    L("built", len(imgs_m), "du2m natives")
    # B0 setup: fixed probes + per-arm standardisation (as run_du2.stage_b0)
    r5 = R.load_r005(); r5.SPLITS = R.SPL["orig"]
    natp, _, _ = r5.pair_indices(df); rtr = r5.real_train_indices(df, natp)
    ms, models = {}, {}
    for a, sub in (("orig", "pilot"), ("fov512", "pilot_fov512"), ("du2", "pilot_du2")):
        F = P.load_features_aligned(df, "primary", "clean", subdir=sub)
        ms[a] = (F[rtr].mean(0).astype(np.float64), np.maximum(F[rtr].std(0).astype(np.float64), 1e-6))
    for a in ("orig", "fov512"):
        models[a] = pickle.loads((R.CACHE / f"b0_{a}" / "models_A0.pkl").read_bytes())
    for cond in ["jpeg75_resize05", "clean"]:   # laundered first (3 of 4 labels); RINE clean skipped (no label)
        Xm = [perturb.apply_condition(imgs_m[s], cond, key=s, seed=0) for s in rows.sha256]
        Xc = [perturb.apply_condition(imgs_c[s], cond, key=s, seed=0) for s in chk.sha256]
        for det in ["communityforensics", "clipdet", "rine"]:
            if det == "rine" and cond == "clean": continue
            fn = R.sp.get_fn(det)
            for arm, X, shas in (("du2chk", Xc, chk.sha256.tolist()), ("du2m", Xm, rows.sha256.tolist())):
                if (det, cond, arm) in done: continue
                t = time.time(); sc = np.asarray(fn(X, batch_size=4), float).reshape(-1)
                out.append(pd.DataFrame(dict(sha256=shas, det=det, model="pretrained", cond=cond, arm=arm, score=sc)))
                pd.concat(out).to_csv(OUTCSV, index=False)
                L(det, cond, arm, len(shas), f"{time.time() - t:.0f}s")
        if ("b0_trained_orig", cond, "du2m") not in done:
            from features import load_backbone, embed_images
            bb = load_backbone(P.SPEC, input_size=224, layers=(-1,), use_cls=True, use_mean=True)
            for arm, X, shas in (("du2chk", Xc, chk.sha256.tolist()), ("du2m", Xm, rows.sha256.tolist())):
                t = time.time(); Fm = embed_images(bb, X, 20, "cpu").astype(np.float32)
                if arm == "du2chk":
                    Fc = P.load_features_aligned(df, "primary", cond, subdir="pilot_du2")[chk.index.to_numpy()]
                    rel = float(np.abs(Fm - Fc).max() / max(1.0, np.abs(Fc).max()))
                    L(f"self-check (iii) B0 features {cond}: max scaled |dF| CPU vs 004 du2 cache = {rel:.2e}")
                for tr in ("orig", "fov512"):
                    Z = ((Fm - ms[tr][0]) / ms[tr][1]).astype(np.float64)
                    for m, mdl in models[tr].items():
                        out.append(pd.DataFrame(dict(sha256=shas, det=f"b0_trained_{tr}", model=m, cond=cond, arm=arm, score=mdl.decision_function(Z))))
                pd.concat(out).to_csv(OUTCSV, index=False)
                L("b0", cond, arm, len(shas), f"{time.time() - t:.0f}s")
    L("done")

if __name__ == "__main__":
    main()
