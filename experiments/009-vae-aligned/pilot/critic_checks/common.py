"""Critic helpers: import 009 run.py as a module (no stage runs), restrict bank to the guard-passed VAEs."""
import importlib.util, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("e2run", ROOT / "experiments/009-vae-aligned/run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
for vn in list(R.VAES):
    if vn != "sd-vae-ft-mse":
        del R.VAES[vn]
import numpy as np, pandas as pd
P = R.P
OUT = Path(__file__).resolve().parent


def arm_data(arm):
    df = R.load_df(arm)
    Xr = {c: P.load_features_aligned(df, "primary", c, subdir=R.SUBDIR[arm]) for c in R.CONDS}
    lab, sp, pool = df.label.to_numpy(), df.split.to_numpy(), df.pool.to_numpy()
    tr = np.where((lab == 0) & (sp == "train"))[0]
    Xrt = {c: R.load_rt_feats(arm, c, df.sha256.to_numpy()[tr]) for c in R.CONDS}
    grp = R.source_groups(df)
    return df, Xr, Xrt, tr, grp


def fitA(Xr, Xrt, tr, grp, rows=None):
    rows = tr if rows is None else rows
    sel = np.isin(tr, rows)
    return R.fit(np.vstack([Xr["clean"][rows], Xrt["clean"][sel]]), np.r_[np.zeros(len(rows)), np.ones(sel.sum())].astype(int),
                 np.concatenate([grp[rows], grp[tr][sel]]))
