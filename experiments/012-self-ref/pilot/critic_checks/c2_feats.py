"""Critic check 2 (CPU, read-only on caches): (a) at clean, are 012's L0 features of t and VAE(t) for the train reals the same as the
features E4 trained the MATCHED probe on (004 laundered cache / 011 dino_rt laundered cache)? i.e. is B at clean literally E4's
matched-pair quantity applied to a test image; (b) size of the B shift vs the between-image spread of d_m; (c) L0 != identity at clean.
Writes critic_checks/c2_feats.json.
  CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/012-self-ref/pilot/critic_checks/c2_feats.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "004-phase4-features"))
sys.path.insert(0, str(ROOT / "experiments" / "lib"))
import p4features as P  # noqa: E402

SUB = {"fov512": None, "orig": None}
sys.path.insert(0, str(ROOT / "experiments" / "009-vae-aligned"))
out = {}


def shards(prefix):
    fs = sorted(f for f in (EXP / "cache" / "feat").glob(prefix + "*.npz") if ".tmp." not in f.name)
    parts = [np.load(f) for f in fs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


for arm, split_csv in (("fov512", "splits_pilot_fov512.csv"), ("orig", "splits_pilot.csv")):
    df = pd.read_csv(ROOT / "experiments" / "004-phase4-features" / "pilot" / split_csv)
    o = out[arm] = {}
    z = shards(f"{arm}__clean__")
    keys = z["keys"].astype(str)
    pos = {k: i for i, k in enumerate(keys)}
    lab = df.set_index("sha256").reindex(keys)
    tr = np.where((lab.label == 0).to_numpy() & (lab.split == "train").to_numpy())[0]
    nat = np.where((lab.label == 0).to_numpy() & (lab.split == "test").to_numpy() & (lab.pool == "vision_native").to_numpy())[0]
    fk = np.where((lab.label == 1).to_numpy())[0]
    # (a1) l0t at clean vs 004's laundered (jpeg75_resize05) secondary features
    sub = "pilot" if arm == "orig" else "pilot_fov512"
    try:
        k4, X4 = P.load_features("secondary", "jpeg75_resize05", subdir=sub)
        p4 = {k: i for i, k in enumerate(k4.astype(str).tolist())}
        ix = [i for i in tr if keys[i] in p4]
        o["l0t_clean_vs_004_laundered"] = dict(n=len(ix), max_abs=float(np.abs(z["l0t"][ix] - X4[[p4[keys[i]] for i in ix]]).max()),
                                               subdir=sub)
    except Exception as e:  # subdir naming may differ
        o["l0t_clean_vs_004_laundered"] = f"unavailable: {e!r}"
    # (a2) l0y at clean vs 011's dino_rt laundered features (E4 matched-probe training round-trips)
    r = np.load(ROOT / "experiments" / "011-vae-trace" / "cache" / "feats" / f"dino_rt__{arm}__jpeg75_resize05.npz")
    p11 = {k: i for i, k in enumerate(r["keys"].astype(str).tolist())}
    ix = [i for i in tr if keys[i] in p11]
    if ix:
        dd = np.abs(z["l0y"][ix] - r["feats"][[p11[keys[i]] for i in ix]])
        o["l0y_clean_vs_011_rt_laundered"] = dict(n=len(ix), max_abs=float(dd.max()), median_abs=float(np.median(dd)))
    # (b) shift size vs spread, with the matched probe
    dm = pickle.loads((EXP / "cache" / "shared" / f"probe__{arm}__matched.pkl").read_bytes()).decision_function
    dp = pickle.loads((EXP / "cache" / "shared" / f"probe__{arm}__primary.pkl").read_bytes()).decision_function
    for cond in ("clean", "jpeg75_resize05"):
        zc = z if cond == "clean" else shards(f"{arm}__{cond}__")
        kc = zc["keys"].astype(str)
        assert (kc == keys).all()
        a_t, a_y = dm(zc["l0t"]), dm(zc["l0y"])
        sh = a_y - a_t
        o[f"B_{cond}"] = dict(
            sd_dm_l0t_natives=float(a_t[nat].std()), sd_dm_l0t_fakes=float(a_t[fk].std()),
            median_shift_natives=float(np.median(sh[nat])), median_shift_fakes=float(np.median(sh[fk])),
            sd_shift_natives=float(sh[nat].std()), sd_shift_fakes=float(sh[fk].std()),
            # Cohen's d of the shift between classes (what B uses) vs of the single-image d_m(L0 t) (what F2m-on-L0 would use)
            d_shift=float((np.mean(sh[nat]) - np.mean(sh[fk])) / np.sqrt((sh[nat].var() + sh[fk].var()) / 2)),
            d_single_l0t=float((np.mean(a_t[fk]) - np.mean(a_t[nat])) / np.sqrt((a_t[nat].var() + a_t[fk].var()) / 2)))
        # (c) L0 is not identity: relative feature change t -> L0 t
        rel = np.linalg.norm(zc["l0t"][nat] - zc["t"][nat], axis=1) / np.linalg.norm(zc["t"][nat], axis=1)
        o[f"L0_rel_feature_change_natives_{cond}"] = float(np.median(rel))
        sha = dp(zc["y"]) - dp(zc["t"])
        o[f"A_{cond}"] = dict(median_shift_natives=float(np.median(sha[nat])), median_shift_fakes=float(np.median(sha[fk])),
                              sd_dp_t_natives=float(dp(zc["t"])[nat].std()))
    o["n"] = dict(train=len(tr), natives=len(nat), fakes=len(fk))
(HERE / "c2_feats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
