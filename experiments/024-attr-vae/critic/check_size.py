"""Critic scratch (CPU, cached features only): native size per generator vs VAE features; argmin preference per generator."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(sys.argv[1]); C = ROOT / "experiments/024-attr-vae/cache"
man = pd.read_csv(ROOT / "data/manifest_raw.csv", keep_default_na=False)
GENS = ["dalle2", "dalle3", "firefly", "midjourney-v5", "stable-diffusion-2", "stable-diffusion-xl", "sd3_large", "flux"]
main = man[(man.label == 1) & man.generator.isin(GENS)][["sha256", "generator", "width", "height"]]
sec = pd.read_csv(ROOT / "experiments/020-attr-foundation/fetch/synthbuster_new_manifest.csv", keep_default_na=False)
sec = sec[sec.generator.isin(["stable-diffusion-1-3", "stable-diffusion-1-4"])][["sha256", "generator", "width", "height"]]
df = pd.concat([main, sec]).set_index("sha256")
def feats(tag, cond, vn):
    z = np.load(C / f"f_{tag}_{cond}_{vn}_cpu.npz", allow_pickle=True)
    return pd.DataFrame(z["feats"], index=z["keys"], columns=[f"{vn[:4]}_{k}" for k in ("e1", "e2", "aedr", "hp8")])
for cond in ["clean", "jpeg75_resize05"]:
    F = pd.concat([pd.concat([feats(t, cond, "sd-vae-ft-mse"), feats(t, cond, "sdxl-vae")], axis=1) for t in ("main", "sec")])
    d = df.join(F, how="inner")
    d["size"] = d.width.astype(str) + "x" + d.height.astype(str)
    d["pref_sd"] = (d["sd-v_aedr"] <= d["sdxl_aedr"]).astype(float)
    d["daedr"] = d["sd-v_aedr"] - d["sdxl_aedr"]
    print(f"\n== {cond}  n={len(d)}")
    g = d.groupby("generator")
    out = g.agg(n=("pref_sd", "size"), sizes=("size", lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().head(3).items())),
                sd_e1=("sd-v_e1", "mean"), xl_e1=("sdxl_e1", "mean"), sd_aedr=("sd-v_aedr", "mean"), xl_aedr=("sdxl_aedr", "mean"),
                sd_hp8=("sd-v_hp8", "mean"), daedr=("daedr", "mean"), pref_sd=("pref_sd", "mean"))
    pd.set_option("display.width", 250); print(out.round(3).to_string())
    # within SD2 / SDXL: does the feature depend on native size?
    for gname in ["stable-diffusion-2", "stable-diffusion-xl"]:
        s = d[d.generator == gname]
        if s["size"].nunique() > 1:
            print(gname, "by size:", s.groupby("size")[["sd-v_e1", "sd-v_aedr", "sdxl_aedr", "pref_sd"]].mean().round(3).to_dict("index"))
