"""Critic check c4: P3 tail - per-device / per-pool FPR of F_b8|T-aug|all ('full') at the pooled 1% calib threshold; calib-vs-test shift."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import run as E
out = {}
for arm in ("orig", "fov512"):
    A = E.Arm(arm, load_feats=False)
    dev = A.df.device.to_numpy()
    o = {}
    for combo in ("F_b8|T-aug|all", "F_last|T-clean|all"):
        S = E.load_scores(arm, "std", tuple(combo.split("|")))[0]
        thr, ncal = E.calib_thresholds(A, S)
        t = thr["full"]
        oc = dict(n_calib=ncal)
        for cond in ("clean", "jpeg75_resize05"):
            s = S["full"][E.CI[cond]]
            per = {}
            for pool in ("vision_native", "vision_whatsapp", "vision_facebook_hq"):
                R = A.idx("test", [pool], label=0)
                per[pool] = {d: int((s[R][dev[R] == d] >= t).sum()) for d in sorted(set(dev[R])) if (s[R][dev[R] == d] >= t).sum() > 0}
            Rr = A.idx("test", ["raise"], label=0)
            Rv = A.idx("test", E.POOLS["vision3"], label=0)
            Cv = A.idx("calib", E.POOLS["vision3"], label=0)
            Cr = A.idx("calib", ["raise"], label=0)
            oc[cond] = dict(fp_by_device=per, raise_fp=int((s[Rr] >= t).sum()),
                            q99=dict(calib_vision=float(np.quantile(s[Cv], .99)), test_vision=float(np.quantile(s[Rv], .99)),
                                     calib_raise=float(np.quantile(s[Cr], .99)), test_raise=float(np.quantile(s[Rr], .99))),
                            thr=float(t))
        o[combo] = oc
    out[arm] = o
(HERE / "c4_fpr_tail.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
