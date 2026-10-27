"""Extra reference (added after the real run; reporting only, no change to the judged pipeline): M0 raw-file metadata baseline for the 14 VISION devices.
Features: width, height, bytes/pixel, JPEG quantisation-table id (EXIF is not used). Random forest, leave-one-photo-out. Native files."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
tab = pd.read_csv(HERE / "results" / "vision_photos.csv")
cfg = json.loads((HERE / "config.json").read_text())
DEVS = cfg["dev"]["devices"]; sets = cfg["dev"]["same_model_sets"]
qt, bpp = [], []
for r in tab.itertuples():
    im = Image.open(ROOT / r.nat)
    qt.append(hash(tuple(tuple(v) for v in im.quantization.values())))
    import os
    bpp.append(os.path.getsize(ROOT / r.nat) / (r.width * r.height))
qid = pd.factorize(pd.Series(qt))[0]
X = np.c_[tab.width, tab.height, bpp, qid]
y = np.array([DEVS.index(d) for d in tab.dev])
pred = np.empty(len(y), int)
for i in range(len(y)):
    m = np.arange(len(y)) != i
    pred[i] = RandomForestClassifier(200, random_state=0).fit(X[m], y[m]).predict(X[i:i + 1])[0]
top1 = float(np.mean([(pred[y == d] == d).mean() for d in range(len(DEVS))]))
dev2set = {d: s for s, ds in sets.items() for d in ds}
res = {"n_quant_tables": int(len(set(qt))), "top1_macro": top1, "chance": 1 / 14}
print(json.dumps(res))
(HERE / "results" / "m0_dev.json").write_text(json.dumps(res, indent=1))
