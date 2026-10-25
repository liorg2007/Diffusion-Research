"""Critic check C2 (H1 official): does the A2-A1 primary gain at jpeg75_resize05 change when the held-out generator's
fakes are restricted to those whose caption id is NOT shared with any train fake of another generator (a fake<->fake
caption-twin path that Amendment 13's mask does not cover)? Reads official caches only.
Usage: env/.venv/bin/python -I c2_caption_share.py <orig|fov512>"""
import sys, re, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[4]
arm = sys.argv[1]
spl = ROOT / ("experiments/003-baselines-eval/splits.csv" if arm == "orig" else "experiments/004-phase4-features/splits_fov512.csv")
df = pd.read_csv(spl, keep_default_na=False, dtype={"vidx": str})
C = ROOT / f"experiments/005-nuisance-projection/official/{arm}/cache"
A1, A2 = np.load(C / "scores_A1.npz"), np.load(C / "scores_A2.npz")
JR = 1   # EVAL_CONDITIONS order: clean, jpeg75_resize05, social_sim, ...
sys.path.insert(0, str(ROOT / "experiments/004-phase4-features")); from p4features import EVAL_CONDITIONS
JR = EVAL_CONDITIONS.index("jpeg75_resize05")
rid = np.array([(re.search(r"(r[0-9a-f]{8,}t)", os.path.basename(p)) or [None, ""])[1] for p in df.src_path])
gen, lab = df.generator.to_numpy(), df.label.to_numpy()
R = np.where((lab == 0) & (df.split.to_numpy() == "test") & (df.pool.to_numpy() == "vision_native"))[0]
def auc(sr, sf):
    r = rankdata(np.concatenate([sr, sf])); return (r[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf))
G = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl", "sd3_large"]
for subset in ("all", "not_shared", "shared"):
    d = []
    for g in G + ["flux"]:
        other = set(rid[(lab == 1) & (gen != g) & df.b0_train.to_numpy()]) - {""}
        F = np.where(gen == g)[0]; sh = np.isin(rid[F], list(other))
        F = F if subset == "all" else (F[~sh] if subset == "not_shared" else F[sh])
        m = "full" if g == "flux" else f"logo_{g}"
        d.append((g, len(F), auc(A1[m][JR][R], A1[m][JR][F]), auc(A2[m][JR][R], A2[m][JR][F])))
    mac = np.mean([x[3] - x[2] for x in d if x[0] != "flux"]); fl = [x for x in d if x[0] == "flux"][0]
    print(f"{arm} {subset:10s}: LOGO-macro A1 {np.mean([x[2] for x in d[:-1]]):.3f} A2-A1 {mac:+.4f} | FLUX n={fl[1]} A1 {fl[2]:.3f} A2-A1 {fl[3]-fl[2]:+.4f}")
