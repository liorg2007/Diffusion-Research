"""Critic check 3 (CPU only, post hoc, descriptive): is the side number (E4's pooled F2 probe, 0.892 clean / 0.665 laundered on the 6
Synthbuster generators, fov512) decoder-specific? Train the SAME probe recipe (lib/probe.fit_probe, balanced, groups = source photo)
on the 297 fov512 train reals vs a Gaussian BLUR of themselves (no VAE), then score the cached phi(t) of the test natives and fakes
exactly as F2p is scored. If the blur-trained probe transfers about as well, F2p is a generic "smoothed vs camera" axis.
DINOv2 runs on CPU (GPU busy). Sanity: our CPU phi of the un-blurred train reals vs 012's cached 't' features.
Writes critic_checks/c3_blur_probe.json.
  CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/012-self-ref/pilot/critic_checks/c3_blur_probe.py
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
ROOT = EXP.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import features  # noqa: E402
import p4features as P  # noqa: E402
import probe  # noqa: E402

torch.set_num_threads(12)
ARM = "fov512"
SIGMAS = [0.867, 2.476]          # 012's sigma_B and sigma_A at clean (fov512, sigma__fov512__clean.json)
t0 = time.time()


def shards(prefix):
    fs = sorted(f for f in (EXP / "cache" / "feat").glob(prefix + "*.npz") if ".tmp." not in f.name)
    parts = [np.load(f) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


df = pd.read_csv(ROOT / "experiments" / "004-phase4-features" / "pilot" / "splits_pilot_fov512.csv").fillna("")
Z = {c: shards(f"{ARM}__{c}__") for c in ("clean", "jpeg75_resize05", "social_sim")}
keys = Z["clean"]["keys"].astype(str)
for c in Z:
    assert (Z[c]["keys"].astype(str) == keys).all()
lab = df.set_index("sha256").reindex(keys)
tr = np.where((lab.label == 0).to_numpy() & (lab.split == "train").to_numpy())[0]
nat = np.where((lab.label == 0).to_numpy() & (lab.split == "test").to_numpy() & (lab.pool == "vision_native").to_numpy())[0]
fk = np.where((lab.label == 1).to_numpy())[0]
gen = lab.generator.to_numpy()
groups = np.array([f"{d}:{v}" if (d != "" and v != "") else s for d, v, s in zip(lab.device, lab.vidx, keys)], dtype=object)
GENS = sorted(set(gen[fk]))

bb = features.load_backbone(P.SPEC, input_size=P.INPUT_SIZE, layers=P.SECONDARY_LAYERS, use_cls=True, use_mean=True)
imgs = []
for i in tr:
    with Image.open(lab.path.iloc[i]) as im:
        im.load()
        imgs.append(im.convert("RGB"))
assert all(im.size == (512, 512) for im in imgs)
# sanity: CPU phi of the plain train reals vs the cached (CUDA) 't' features
f_plain = features.embed_images(bb, imgs[:20], 20, "cpu").astype(np.float32)
sanity = float(np.abs(f_plain - Z["clean"]["t"][tr[:20]]).max())
print(f"sanity max|dphi| (20 train reals, CPU vs 012 cache) = {sanity:.2e}  ({time.time() - t0:.0f}s)", flush=True)


def macro(s_nat, s_fk):
    return float(np.mean([mannwhitneyu(s_fk[gen[fk] == g], s_nat, alternative="two-sided").statistic /
                          ((gen[fk] == g).sum() * len(s_nat)) for g in GENS]))


out = dict(arm=ARM, sanity_max_abs_dphi_cpu_vs_cache=sanity, n_train=len(tr), n_natives=len(nat), n_fakes=len(fk), sigmas={})
# reference: E4 primary probe (refit by 012, shared cache) scored the same way
dp = pickle.loads((EXP / "cache" / "shared" / f"probe__{ARM}__primary.pkl").read_bytes()).decision_function
out["F2p_reference"] = {c: macro(dp(Z[c]["t"][nat]), dp(Z[c]["t"][fk])) for c in Z}
Xp = Z["clean"]["t"][tr]
for sg in SIGMAS:
    fb = features.embed_images(bb, [im.filter(ImageFilter.GaussianBlur(sg)) for im in imgs], 20, "cpu").astype(np.float32)
    X = np.vstack([Xp, fb])
    y = np.r_[np.zeros(len(tr), int), np.ones(len(tr), int)]
    m = probe.fit_probe(X, y, 0, groups=np.concatenate([groups[tr], groups[tr]]), class_weight="balanced")
    d = m.decision_function
    r = {c: macro(d(Z[c]["t"][nat]), d(Z[c]["t"][fk])) for c in Z}
    r["per_generator_clean"] = {g: float(mannwhitneyu(d(Z["clean"]["t"][fk][gen[fk] == g]), d(Z["clean"]["t"][nat]),
                                                      alternative="two-sided").statistic / ((gen[fk] == g).sum() * len(nat))) for g in GENS}
    r["C"] = float(m[-1].C_[0])
    # how aligned is the blur direction with the VAE direction? (cosine of standardized-space weights)
    wb = m[-1].coef_[0] / m[0].scale_
    pr = pickle.loads((EXP / "cache" / "shared" / f"probe__{ARM}__primary.pkl").read_bytes())
    wv = pr[-1].coef_[0] / pr[0].scale_
    r["cos_raw_weights_blur_vs_vae_probe"] = float(wb @ wv / np.linalg.norm(wb) / np.linalg.norm(wv))
    r["spearman_scores_vs_F2p_on_natives_plus_fakes_clean"] = float(pd.Series(d(Z["clean"]["t"][np.r_[nat, fk]])).corr(
        pd.Series(dp(Z["clean"]["t"][np.r_[nat, fk]])), method="spearman"))
    out["sigmas"][str(sg)] = r
    print(f"sigma {sg}: blur-trained probe macro {({k: round(v, 3) for k, v in r.items() if k in Z})}; cos {r['cos_raw_weights_blur_vs_vae_probe']:.2f}; "
          f"rho vs F2p {r['spearman_scores_vs_F2p_on_natives_plus_fakes_clean']:.2f} ({time.time() - t0:.0f}s)", flush=True)
print("F2p reference", {k: round(v, 3) for k, v in out["F2p_reference"].items()})
(HERE / "c3_blur_probe.json").write_text(json.dumps(out, indent=1))
