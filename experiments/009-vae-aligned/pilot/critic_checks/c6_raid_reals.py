"""Descriptive, not pre-registered: A (orig arm, SD-VAE aligned only) and B0 (A0 on all pilot train data, refit here with the
same recipe) on RAID ORIGINAL images (non-VISION reals from LAION/D3 per kb, JPEG ~q75; fakes = 4 RAID generators), 256 center
crop (RAID's preprocessing), no padding (Amendment 9 clar. 2 primary rule), DINOv2 on CPU. Output: c6_raid_reals.json"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch; torch.set_num_threads(6)
from PIL import Image
from torchvision import transforms as T
import features
man = pd.read_csv(ROOT / "experiments/010-raid-transfer/raid_manifest.csv")
man = man[man.kind == "original"].reset_index(drop=True)
imgs, keep = [], []
for i, r in man.iterrows():
    with Image.open(ROOT / r.path) as im:
        im.load()
        if min(im.size) < 256:
            continue
        imgs.append(T.CenterCrop(256)(im.convert("RGB"))); keep.append(i)
man = man.loc[keep].reset_index(drop=True)
bb = R.p4run.secondary_backbone()
F = features.embed_images(bb, imgs, 8, "cpu")[:, -1536:]
df, Xr, Xrt, tr, grp = arm_data("orig")
mA = fitA(Xr, Xrt, tr, grp)
b0 = df.b0_train.to_numpy()
mB = R.fit(Xr["clean"][b0], df.label.to_numpy()[b0], grp[b0])
nat = np.where((df.label == 0) & (df.split == "test") & (df.pool == "vision_native"))[0]
out = dict(n=man.cls.value_counts().to_dict())
for nm, m in (("A", mA), ("B0_full", mB)):
    s = m.decision_function(F)
    y = (man.cls != "real").to_numpy()
    sv = m.decision_function(Xr["clean"][nat])
    thr = np.quantile(sv, 0.99)
    out[nm] = dict(auroc_per_gen={g: R.auc(s[~y], s[(man.cls == g).to_numpy()]) for g in sorted(set(man.cls) - {"real"})},
                   auroc_pooled=R.auc(s[~y], s[y]),
                   frac_raid_reals_flagged_at_vision_1pctFPR=float((s[~y] > thr).mean()),
                   median_raid_real=float(np.median(s[~y])), median_vision_test_native=float(np.median(sv)),
                   auroc_vision_natives_vs_raid_reals_fake_high=R.auc(sv, s[~y]))
(OUT / "c6_raid_reals.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
