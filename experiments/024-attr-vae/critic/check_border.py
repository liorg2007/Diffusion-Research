"""Critic scratch (CPU, sd-vae only, clean): does SD1.x's off-manifold AEDR come from the full-frame border? Interior 448 crop at (32,32)
(no image border, 8-px phase kept) vs cached top-left 512, for 8 SD1.4 and 8 SD2 images."""
import sys, os, importlib.util, numpy as np
from pathlib import Path
from PIL import Image
HERE = Path(sys.argv[1]); os.environ["NTHREADS"] = "16"
spec = importlib.util.spec_from_file_location("r024", HERE / "run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
main, sec, _ = R.load_tables()
rng = np.random.default_rng(1)
picks = {"stable-diffusion-1-4": sec[sec.generator == "stable-diffusion-1-4"], "stable-diffusion-2": main[main.generator == "stable-diffusion-2"]}
dev = R.get_dev("cpu"); vae = R.load_vae("sd-vae-ft-mse", dev)
for g, d in picks.items():
    d = d.iloc[rng.choice(len(d), 8, replace=False)]
    arrs = []
    for p in d.path:
        im = Image.open(R.ROOT / p); im.load(); a = np.asarray(im.convert("RGB"))
        arrs.append(np.ascontiguousarray(a[32:480, 32:480]))
    F, _ = R.features_for(vae, arrs, R.Batcher(4))
    print(g, "interior448 log_aedr per image:", np.round(F[:, 2], 3).tolist(), "mean aedr", round(float(np.exp(F[:, 2]).mean()), 3),
          "| mean log_e1", round(float(F[:, 0].mean()), 3))
