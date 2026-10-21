"""A deliberately dumb baseline: if file/format statistics alone separate real from fake, the dataset leaks."""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import compute_metrics
from preprocess import jpeg_quality_estimate

FEATURE_NAMES = ["bytes_per_pixel", "has_quant_table", "jpeg_quality", "width", "height", "hf_energy"]


def hf_energy(gray):
    """Mean |x - 3x3 box blur| of a float grayscale array."""
    p = np.pad(gray, 1, mode="edge")
    blur = sum(p[i:i + gray.shape[0], j:j + gray.shape[1]] for i in range(3) for j in range(3)) / 9.0
    return float(np.abs(gray - blur).mean())


def format_features(path, sidecar_row=None):
    """File-level stats. If a sidecar row (from preprocess) is given, file-format stats describe the
    ORIGINAL file (that is where leakage lives); hf_energy is always measured on the pixels at `path`."""
    with Image.open(path) as im:
        w, h = im.size
        q = jpeg_quality_estimate(im)
        gray = np.asarray(im.convert("L"), dtype=np.float32)
    nbytes = None
    if sidecar_row is not None:
        w, h, nbytes = sidecar_row["orig_width"], sidecar_row["orig_height"], sidecar_row["orig_bytes"]
        q = sidecar_row["orig_jpeg_quality"]
        q = None if pd.isna(q) else q
    if nbytes is None:
        import os
        nbytes = os.path.getsize(path)
    return [nbytes / (w * h), float(q is not None), float(q or 0), float(w), float(h), hf_energy(gray)]


def format_bias_report(df, sidecar=None, seed=0, n_boot=200):
    """Train logistic regression on split=='train', report AUROC on split=='test'.
    High AUROC (>~0.6) means format/size/high-frequency statistics leak the label."""
    side = {} if sidecar is None else sidecar.set_index("sha256").to_dict("index")
    X = np.array([format_features(r.path, side.get(r.sha256)) for r in df.itertuples()])
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    y = df.label.to_numpy()
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return dict(auroc=float("nan"), note="train or test missing a class")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    clf.fit(X[tr], y[tr])
    s = clf.predict_proba(X[te])[:, 1]
    m = compute_metrics(y[te], s, n_boot=n_boot, seed=seed)
    coef = dict(zip(FEATURE_NAMES, clf[-1].coef_[0].round(3).tolist()))
    return dict(auroc=m["auroc"]["value"], auroc_ci=[m["auroc"]["ci_lo"], m["auroc"]["ci_hi"]],
                n_train=int(tr.sum()), n_test=int(te.sum()), standardized_coef=coef,
                used_sidecar=bool(side), leakage_flag=bool(m["auroc"]["ci_lo"] > 0.6))
