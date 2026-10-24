"""Critic check 3: fragility of the 43-calib-native (max+eps) thresholds; cross-zoom FPR under alternative thresholds.
Loads E1's cached B0 models (no fitting: models_A0.pkl + scores_A0.npz exist) via run.stage_b0 on CPU. Writes nothing."""
import importlib.util, sys
import numpy as np, pandas as pd
ROOT = "/home/zera/project/Diffusion-Research"
spec = importlib.util.spec_from_file_location("e1", ROOT + "/experiments/008-scale-sensitivity/run.py")
e1 = importlib.util.module_from_spec(spec); sys.modules["e1"] = e1; spec.loader.exec_module(e1)
import metrics
df = e1.load_df("orig")
B0, B0X, chk = e1.stage_b0()
d = pd.read_csv(ROOT + "/experiments/008-scale-sensitivity/pilot/scores_long.csv.gz"); d = d[d.model == "pretrained"]
sha2i = {s: i for i, s in enumerate(df.sha256)}
n = len(df); lab, pool, split, dev = df.label.values, df.pool.values, df.split.values, df.device.values
sel = e1.selection(df)
calN = np.where(sel & (lab == 0) & (split == "calib") & (pool == "vision_native"))[0]
calA = np.where(sel & (lab == 0) & (split == "calib"))[0]
teN = np.where(sel & (lab == 0) & (split == "test") & (pool == "vision_native"))[0]
fk = np.where(sel & (split == "test") & (lab == 1))[0]
print("calib natives:", len(calN), "devices:", len(set(dev[calN])), "| test natives", len(teN), "devices", len(set(dev[teN])),
      "| calib/test device overlap", len(set(dev[calN]) & set(dev[teN])))
def V(det, a, c, g=None, tr=None):
    ci = e1.CONDS.index(c)
    if det == "b0x": return np.asarray(B0X[tr][a][f"logo_{g}"][ci], float)
    v = np.full(n, np.nan); q = d[(d.detector == det) & (d.arm == a) & (d.condition == c)]
    v[[sha2i[s] for s in q.sha256]] = q.score.values; return v
rng = np.random.default_rng(0)
cdev = dev[calN]; ud = np.unique(cdev)
def boot_thr(s, B=2000):
    """device-cluster bootstrap of the 43 calib natives -> distribution of the max+eps threshold."""
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(ud, len(ud)); ix = np.concatenate([np.where(cdev == u)[0] for u in pick])
        out[b] = metrics.threshold_for_fpr(s[calN][ix], 0.01)
    return out
cases = [("b0x", "orig", "fov512", "orig"), ("b0x", "fov512", "orig", "fov512"),
         ("communityforensics", "fov512", "orig", None), ("communityforensics", "orig", "fov512", None),
         ("rine", "orig", "fov512", None), ("rine", "fov512", "orig", None), ("clipdet", "orig", "fov512", None)]
for c in e1.CONDS:
    print(f"\n#### {c}")
    for det, src, tgt, tr in cases:
        gens = e1.GENS if det == "b0x" else [None]
        R = {k: [] for k in ["thr_max", "thr_2nd", "thr_3rd", "within_max", "cross_max", "cross_2nd", "cross_3rd", "cross_all129",
                             "cross_at_within1", "cross_at_within2", "cross_at_within5", "boot_cross_med", "boot_cross_lo", "boot_cross_hi"]}
        for g in gens:
            ss, st = V(det, src, c, g, tr), V(det, tgt, c, g, tr)
            top = np.sort(ss[calN])[::-1]
            for k, j in (("max", 0), ("2nd", 1), ("3rd", 2)):
                t = np.nextafter(top[j], np.inf); R[f"cross_{k}"].append((st[teN] >= t).mean())
                if k == "max": R["within_max"].append((ss[teN] >= t).mean()); R["thr_max"].append(t)
                else: R[f"thr_{k}"].append(t)
            R["cross_all129"].append((st[teN] >= metrics.threshold_for_fpr(ss[calA], 0.01)).mean())
            for q in (1, 2, 5):   # threshold set on the SOURCE arm's TEST natives (oracle, descriptive): within FPR = q%
                t = np.quantile(ss[teN], 1 - q / 100); R[f"cross_at_within{q}"].append((st[teN] >= t).mean())
            bt = boot_thr(ss, 1000); fr = np.array([(st[teN] >= t).mean() for t in bt])
            R["boot_cross_med"].append(np.median(fr)); R["boot_cross_lo"].append(np.percentile(fr, 2.5)); R["boot_cross_hi"].append(np.percentile(fr, 97.5))
        m = {k: float(np.mean(v)) for k, v in R.items()}
        name = f"{det}{'(trained '+tr+')' if tr else ''} thr:{src}->test:{tgt}"
        print(f"{name:52s} within(max)={m['within_max']:.3f} cross: max={m['cross_max']:.3f} 2nd={m['cross_2nd']:.3f} 3rd={m['cross_3rd']:.3f} "
              f"all129={m['cross_all129']:.3f} | devboot median {m['boot_cross_med']:.3f} [{m['boot_cross_lo']:.3f},{m['boot_cross_hi']:.3f}] | "
              f"oracle within 1/2/5% -> cross {m['cross_at_within1']:.3f}/{m['cross_at_within2']:.3f}/{m['cross_at_within5']:.3f}")
# where do the B0-fixed-probe test-native scores move? median shift in units of the source-arm native IQR
print("\nB0 fixed probe (trained orig): median score of test natives / fakes, orig vs fov512 features")
for c in e1.CONDS:
    for g in e1.GENS:
        so, sf = V("b0x", "orig", c, g, "orig"), V("b0x", "fov512", c, g, "orig")
        print(c, g, "natives", round(np.median(so[teN]), 2), "->", round(np.median(sf[teN]), 2),
              "| calib-nat max", round(np.max(so[calN]), 2), "2nd", round(np.sort(so[calN])[-2], 2),
              "| fakes(g)", round(np.median(so[fk][df.generator.values[fk] == g]), 2), "->", round(np.median(sf[fk][df.generator.values[fk] == g]), 2))
