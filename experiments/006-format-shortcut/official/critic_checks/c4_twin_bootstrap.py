"""Critic check 4 (CPU): B0 macro AUROC of RAISE test reals split by caption-twin status (fake filenames = RAISE ids),
2x2 (twin among B0-train fakes) x (twin among test fakes), with an image-level bootstrap CI (reals by image, fakes by image per
generator, 2000 draws, seed 0). The direct twin pair is at most 1 of ~60 fakes per generator, so it can move a real's AUROC by
<= ~0.017; larger differences point to the probe having learned the twin's CONTENT from training."""
import importlib.util, re, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("ro", ROOT / "experiments/006-format-shortcut/run_official.py")
ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
import h4
OUT = Path(__file__).resolve().parent
L = []
for arm, sp_, sub in [("orig", "experiments/003-baselines-eval/splits.csv", None),
                      ("fov512", "experiments/004-phase4-features/splits_fov512.csv", "fov512")]:
    df = ro.load_df(ROOT / sp_)
    X = ro.load_aligned_multi(df, "clean", [sub])
    gens_all = ro.MAIN + ro.TEST_ONLY
    B = ro.apply_b0(ro.fit_b0(df, X, gens_all, strict=True), X, gens_all)
    lab, gen, pool, sp = df.label.values, df.generator.values, df.pool.values, df.split.values
    rid = df.src_path.map(lambda p: (re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1]).values
    trn = set(rid[(lab == 1) & df.b0_train.values]); tst = set(rid[(lab == 1) & (sp == "test") & np.isin(gen, ro.MAIN)])
    R = np.where((sp == "test") & (pool == "raise"))[0]
    a = np.array([rid[r] in trn for r in R]); b = np.array([rid[r] in tst for r in R])
    fs = h4.FakeSet(gen, ro.MAIN, (sp == "test") & (lab == 1), 2000, 0)
    dev = np.array([""] * len(df), dtype=object); sha = df.sha256.values
    def est(mask, name):
        rs = h4.RealSet(R[mask], dev, sha, 2000, 0, name=name)
        p, bt = h4.macro_auroc(lambda g: B[g], rs, fs)
        return p, bt
    cells = {"train&test": a & b, "train only": a & ~b, "test only": ~a & b, "neither": ~a & ~b, "any twin": a | b}
    res = {k: est(m, k) for k, m in cells.items() if m.sum()}
    for k, (p, bt) in res.items():
        L.append(f"[{arm}] RAISE twin {k:11s} n={cells[k].sum():3d}: B0 AUROC {p:.3f} [{np.percentile(bt,2.5):.3f},{np.percentile(bt,97.5):.3f}]")
    p, bt = res["neither"][0] - res["any twin"][0], res["neither"][1] - res["any twin"][1]
    L.append(f"[{arm}] neither - any twin: {p:+.3f} [{np.percentile(bt,2.5):+.3f},{np.percentile(bt,97.5):+.3f}] (reals independent, fakes paired)")
(OUT / "c4_twin_bootstrap.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
