"""Critic check 2 (CPU, cached 004 features only): why does B0's VISION - RAISE clean gap persist in fov512?
(a) content proximity: each test real's max cosine similarity (B0 primary features, L2-normalised) to the 1,020 fakes of the 7 MAIN generators
    (train+test); RAISE reals' caption twins are among the fakes (fake filenames = RAISE ids).
(b) B0 AUROC per pool within bins of that similarity (content-matched comparison).
(c) RAISE reals whose caption-twin fake is in TEST vs not.
(d) raw image sizes per pool (resize history)."""
import importlib.util, re, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("ro", ROOT / "experiments/006-format-shortcut/run_official.py")
ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
import h4
from scipy.stats import spearmanr
OUT = Path(__file__).resolve().parent
L = []
man = pd.read_csv(ROOT / "data/manifest_raw.csv")
for arm, sp_, sub in [("orig", "experiments/003-baselines-eval/splits.csv", None),
                      ("fov512", "experiments/004-phase4-features/splits_fov512.csv", "fov512")]:
    df = ro.load_df(ROOT / sp_)
    X = ro.load_aligned_multi(df, "clean", [sub])
    gens_all = ro.MAIN + ro.TEST_ONLY
    fit = ro.fit_b0(df, X, gens_all, strict=True)
    B = ro.apply_b0(fit, X, gens_all)
    lab, gen, pool, sp = df.label.values, df.generator.values, df.pool.values, df.split.values
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    fk_all = np.where((lab == 1) & np.isin(gen, ro.MAIN))[0]
    rid = df.src_path.map(lambda p: (re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1]).values
    test_fake_ids = set(rid[(lab == 1) & (sp == "test") & np.isin(gen, ro.MAIN)])
    nn = {}
    for pk in ["vision_native", "raise"]:
        R = np.where((sp == "test") & (pool == pk))[0]
        S = Xn[R] @ Xn[fk_all].T
        # exclude a real's own caption twins so 'content proximity' is not just the twin
        if pk == "raise":
            for j, r in enumerate(R):
                S[j, rid[fk_all] == rid[r]] = -1
        nn[pk] = (R, S.max(1), np.sort(S, 1)[:, -5:].mean(1))
        L.append(f"[{arm}] {pk}: n={len(R)} max-cos to fakes (twins excluded) median {np.median(nn[pk][1]):.3f} IQR "
                 f"[{np.percentile(nn[pk][1],25):.3f},{np.percentile(nn[pk][1],75):.3f}]; top5-mean median {np.median(nn[pk][2]):.3f}")
    # B0 macro AUROC per pool (test fakes MAIN) as a function of content proximity
    FT = {g: np.where((sp == "test") & (lab == 1) & (gen == g))[0] for g in ro.MAIN}
    def mac(Ridx):
        return np.mean([h4.auroc_point(B[g][Ridx], B[g][FT[g]]) for g in ro.MAIN])
    allsim = np.concatenate([nn["vision_native"][2], nn["raise"][2]])
    edges = np.quantile(allsim, [0, .25, .5, .75, 1.0])
    for pk in ["vision_native", "raise"]:
        R, _, s5 = nn[pk]
        L.append(f"[{arm}] B0 macro AUROC {pk}: all {mac(R):.3f}")
        for b in range(4):
            m = (s5 >= edges[b]) & (s5 <= edges[b + 1])
            if m.sum() >= 10:
                L.append(f"      sim-quartile {b+1} [{edges[b]:.3f},{edges[b+1]:.3f}] n={m.sum():3d} AUROC {mac(R[m]):.3f}")
        sc = np.mean([B[g][R] for g in ro.MAIN], 0)
        rho = spearmanr(sc, s5).correlation
        L.append(f"      Spearman(B0 mean logit of reals, content proximity) = {rho:+.3f}")
    R = nn["raise"][0]
    tw = np.array([rid[r] in test_fake_ids for r in R])
    L.append(f"[{arm}] RAISE test reals with a caption-twin TEST fake: n={tw.sum()} AUROC {mac(R[tw]):.3f}; without: n={(~tw).sum()} AUROC {mac(R[~tw]):.3f}")
    train_fake_ids = set(rid[(lab == 1) & df.b0_train.values])
    tt = np.array([rid[r] in train_fake_ids for r in R])
    L.append(f"[{arm}] RAISE test reals with a caption-twin B0-TRAIN fake: n={tt.sum()} AUROC {mac(R[tt]):.3f}; without: n={(~tt).sum()} AUROC {mac(R[~tt]):.3f}")
    nt = (~tw) & (~tt)
    L.append(f"[{arm}] RAISE test reals with NO twin in test or train fakes: n={nt.sum()} AUROC {mac(R[nt]):.3f}")
    for pk in ["vision_whatsapp", "vision_facebook_hq"]:
        Rp = np.where((sp == "test") & (pool == pk))[0]
        L.append(f"[{arm}] B0 macro AUROC {pk} (scale-matched control for RAISE, same photos as natives): {mac(Rp):.3f}")
# (d) sizes
d0 = ro.load_df(ROOT / "experiments/003-baselines-eval/splits.csv")
mm = man.set_index("sha256")
for pk in ["vision_native", "raise"]:
    r = d0[(d0.pool == pk) & (d0.split == "test")]
    mn = r.src_path.map(lambda p: None)
    m2 = man[man.path.isin(r.src_path)]
    ss = np.minimum(m2.width, m2.height)
    L.append(f"raw short side {pk}: median {np.median(ss):.0f} (min {ss.min()}, max {ss.max()}), formats {m2.orig_format.value_counts().to_dict()}")
for g in ro.MAIN + ["flux"]:
    m2 = man[man.generator == g]
    ss = np.minimum(m2.width, m2.height)
    L.append(f"raw short side {g}: median {np.median(ss):.0f}, formats {m2.orig_format.value_counts().to_dict()}")
(OUT / "c2_vision_raise_gap.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
