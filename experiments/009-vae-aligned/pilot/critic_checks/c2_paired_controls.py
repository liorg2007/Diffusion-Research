"""Q1 controls on UNSEEN-device test natives (orig arm), all on CPU (sd-vae-ft-mse, the guard-passed VAE; DINOv2 CPU):
  resave = native crop re-saved as JPEG at its source QF (no VAE)  -> does A flag a fresh JPEG alone?
  vae_png = VAE round-trip saved PNG (no fresh JPEG)              -> VAE trace alone
  vae_jpg = VAE round-trip re-saved JPEG at source QF (= how aligned fakes were made)
A's paired AUROC (variant vs its own native), clean and jpeg75_resize05. Also out-of-fold (device-grouped) AUROC of
reals vs their own round-trips on the TRAIN pairs. Output: c2_paired_controls.json"""
import sys, os, io, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from PIL import Image
from preprocess import jpeg_quality_estimate
from perturb import apply_condition
import features
torch.set_num_threads(16)
N = int(os.environ.get("N_CTRL", "80"))
df, Xr, Xrt, tr, grp = arm_data("orig")
lab, sp, pool, gen = (df[c].to_numpy() for c in ("label", "split", "pool", "generator"))
mA = fitA(Xr, Xrt, tr, grp)
nat = np.where((lab == 0) & (sp == "test") & (pool == "vision_native"))[0]
rng = np.random.default_rng(1)
pick = np.sort(rng.choice(nat, N, replace=False))
cache = OUT / "c2_cache.npz"
if cache.exists():
    z = np.load(cache); V = {k: z[k] for k in z}
else:
    vae = R.load_vae("sd-vae-ft-mse", torch.device("cpu"))
    imgs = [np.asarray(Image.open(df.path.iloc[i]).convert("RGB")) for i in pick]
    V = {"native": np.stack(imgs)}
    V["vae"] = np.concatenate([R.q8(R.recon_f(vae, V["native"][j:j + 2])) for j in range(0, N, 2)])
    np.savez(cache, **V)
def jpg(a, q):
    b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=int(q)); b.seek(0); return Image.open(b).convert("RGB")
qf = []
for i in pick:
    with Image.open(df.src_path.iloc[i]) as im:
        qf.append(jpeg_quality_estimate(im))
var = {"native": [Image.fromarray(a) for a in V["native"]],
       "resave": [jpg(a, q) for a, q in zip(V["native"], qf)],
       "vae_png": [Image.fromarray(a) for a in V["vae"]],
       "vae_jpg": [jpg(a, q) for a, q in zip(V["vae"], qf)]}
bb = R.p4run.secondary_backbone()
h = 1536
out = dict(n=N, qf_median=float(np.median(qf)))
for c in R.CONDS:
    F = {k: features.embed_images(bb, [apply_condition(im, c, key=df.sha256.iloc[i], seed=R.P.SEED) for im, i in zip(v, pick)], 8, "cpu")[:, -h:]
         for k, v in var.items()}
    s = {k: mA.decision_function(f) for k, f in F.items()}
    out[c] = {"cpu_vs_cached_native_maxdlogit": float(np.abs(s["native"] - mA.decision_function(Xr[c][pick])).max())}
    for k in ("resave", "vae_png", "vae_jpg"):
        out[c][k] = dict(paired_auroc=R.auc(s["native"], s[k]), frac_score_up=float((s[k] > s["native"]).mean()),
                         median_dscore=float(np.median(s[k] - s["native"])))
    fak = np.where(lab == 1)[0]
    sa = mA.decision_function(Xr[c])
    out[c]["ref_macro_auroc_natives_subset_vs_fakes"] = float(np.mean([R.auc(sa[pick], sa[fak[gen[fak] == g]]) for g in R.GENS]))
    out[c]["ref_median_score_natives"] = float(np.median(s["native"]))
# OOF on train pairs, grouped by DEVICE (stricter than device:vidx)
from sklearn.model_selection import GroupKFold
dev = df.device.to_numpy()
X0, X1 = Xr["clean"][tr], Xrt["clean"]
oof = {c: np.zeros(2 * len(tr)) for c in R.CONDS}
for a, b in GroupKFold(5).split(tr, groups=dev[tr]):
    m = R.fit(np.vstack([X0[a], X1[a]]), np.r_[np.zeros(len(a)), np.ones(len(a))].astype(int), np.r_[grp[tr][a], grp[tr][a]])
    for c in R.CONDS:
        oof[c][b] = m.decision_function(Xr[c][tr][b]); oof[c][len(tr) + b] = m.decision_function(Xrt[c][b])
out["train_pairs_oof_device_grouped_auroc"] = {c: R.auc(oof[c][:len(tr)], oof[c][len(tr):]) for c in R.CONDS}
(OUT / "c2_paired_controls.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
