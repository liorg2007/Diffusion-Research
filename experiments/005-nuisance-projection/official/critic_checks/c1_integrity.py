"""Critic check C1 (H1 official): leakage + cache-validity checks. Writes nothing into official/ except stdout.
Usage: env/.venv/bin/python -I c1_integrity.py <orig|fov512> <scratch_cache_dir>"""
import sys, re, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/005-nuisance-projection"))
import run
arm = sys.argv[1]; scratch = Path(sys.argv[2])
sub, spl = ("feats", ROOT/"experiments/003-baselines-eval/splits.csv") if arm == "orig" else ("fov512", ROOT/"experiments/004-phase4-features/splits_fov512.csv")
a = run.parse(["--subdir", sub, "--splits", str(spl), "--out-dir", str(scratch), "--label", "CRITIC",
               "--generators", *"dalle2 dalle3 firefly midjourney-v5 stable-diffusion-2 stable-diffusion-xl sd3_large".split(),
               "--heldout-generators", "flux", "--grouped-cv", "--balanced"])
run.configure(a)
df, D = run.load_data(("primary",))
nat, wa, fb = run.pair_indices(df)
rtr = run.real_train_indices(df, nat)
# 1. leakage: every row feeding mu/sd and the projector deltas is a TRAIN REAL; FLUX never in b0_train
used = np.unique(np.concatenate([nat, wa, fb, rtr]))
print("projector/standardisation rows:", len(used), "all train reals:",
      bool(((df.label.to_numpy()[used] == 0) & (df.split.to_numpy()[used] == "train")).all()))
print("FLUX rows in b0_train:", int((df.b0_train & (df.generator == "flux")).sum()),
      "| FLUX splits:", df.split[df.generator == "flux"].value_counts().to_dict())
print("calib/test rows in b0_train:", int((df.b0_train & df.split.isin(["calib", "test"])).sum()))
# 2. twin mask: size, and fake<->fake caption sharing across generators (not covered by Amendment 13)
rid = np.array([(re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1] for p in df.src_path])
m = run.twin_drop_mask(df)
print("twin-drop mask:", int(m.sum()), "of", int(((df.label == 1) & (df.split == "train")).sum()), "train fakes")
fk = (df.label == 1).to_numpy() & (rid != "")
for g in run.GENS + ["flux"]:
    G = fk & (df.generator.to_numpy() == g)
    other_train = set(rid[fk & (df.generator.to_numpy() != g) & df.b0_train.to_numpy()])
    print(f"  {g}: {G.sum()} fakes with caption id; share id with a train fake of another generator: "
          f"{np.isin(rid[G], list(other_train)).mean():.2f}")
# 3. cache validity: refit A1/A2 (full + logo_sd3_large) on CPU from scratch, compare to the official cache
F = D["primary"]
mu = F["clean"][rtr].mean(0).astype(np.float64); sd = np.maximum(F["clean"][rtr].std(0).astype(np.float64), 1e-6)
zf = lambda X: ((X - mu) / sd).astype(np.float64)
Zev = {c: zf(F["ev"][c]) for c in run.CONDS}; Zg = {c: zf(F["grid"][c]) for c in run.GRID_CONDITIONS}
Dl, _, _ = run.build_deltas({"clean": Zev["clean"]}, Zg, nat, wa, fb, grid_reals=rtr)
w, V = run.eig_second_moment(Dl, False); ks, _ = run.kstar(w)
print("k* recomputed:", ks)
run.MODELS = ["full", "logo_sd3_large"]; run.CACHE = scratch; scratch.mkdir(parents=True, exist_ok=True)
off = ROOT / f"experiments/005-nuisance-projection/official/{arm}/cache"
for name, T in (("A1", lambda X: X), ("A2", lambda X: run.project(X, V[:, :ks]))):
    p = scratch / f"scores_{name}.npz"
    if p.exists(): p.unlink()
    S = run.run_arm(name, df, T, Zev, Zg, aug=True)
    O = np.load(off / f"scores_{name}.npz")
    for mdl in run.MODELS:
        print(f"{name}/{mdl}: max|refit - official cache| = {np.abs(S[mdl] - O[mdl]).max():.2e}")
