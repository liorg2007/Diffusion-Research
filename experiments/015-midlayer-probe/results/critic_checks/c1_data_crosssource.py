"""Critic check c1: split facts, cross-source diagnosis (FPR blow-up vs missed fakes), per-source thresholds. Read-only on E6 caches."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np, pandas as pd
import run as E
from sklearn.metrics import roc_auc_score

out = {}
for arm in ("orig", "fov512"):
    A = E.Arm(arm, load_feats=False)
    df = A.df
    o = {}
    fk = A.lab == 1
    o["fakes_with_caption_id_frac"] = float((A.rid[fk] != "").mean())
    o["fake_source_dataset"] = df.source_dataset[fk].value_counts().to_dict()
    o["real_source_dataset"] = df[~fk].groupby(["pool", "source_dataset"]).size().astype(int).to_dict().__repr__()
    o["roles_by_pool_label"] = pd.crosstab(A.pool + "|" + A.lab.astype(str), A.role).to_dict()
    trdev = set(df.device[(A.role == "train") & (A.pool == "vision_native")])
    cadev = set(df.device[(A.role == "calib") & (A.pool == "vision_native")])
    tedev = set(df.device[(A.role == "test") & (A.pool == "vision_native")])
    o["n_dev"] = dict(train=len(trdev), calib=len(cadev), test=len(tedev), calib_in_train=len(cadev & trdev), test_in_train=len(tedev & trdev))
    calib = A.idx("calib", E.POOLS["allreals"], label=0)
    o["calib_reals_by_pool"] = pd.Series(A.pool[calib]).value_counts().to_dict()
    # b0_train overlap with E6 roles
    b0 = df.b0_train.to_numpy()
    o["b0_train_rows_by_E6_role"] = pd.Series(A.role[b0]).value_counts().to_dict()
    ev_ids = set(A.rid[(np.isin(A.role, ["calib", "test"])) & (A.rid != "")])
    o["b0_train_rows_sharing_id_with_E6_eval"] = int(sum(1 for i in np.where(b0)[0] if A.rid[i] in ev_ids))

    ci = E.CI["jpeg75_resize05"]
    Rv = A.idx("test", ["vision_native", "vision_whatsapp", "vision_facebook_hq"], label=0)
    Rn = A.idx("test", ["vision_native"], label=0)
    Rr = A.idx("test", ["raise"], label=0)
    Cr = A.idx("calib", ["raise"], label=0)
    Cv = A.idx("calib", ["vision_native"], label=0)
    F = A.idx("test", ["fake"], label=1)
    diag = {}
    for combo in ("F_b8|T-aug|raise_only", "F_b8|T-aug|vision_only", "F_b8|T-aug|all", "F_last|T-aug|raise_only",
                  "F_last|T-clean|all", "F_b8|T-clean|raise_only"):
        S, _, _ = E.load_scores(arm, "std", tuple(combo.split("|")))
        d = {}
        for cname in ("clean", "jpeg75_resize05"):
            s = S["full"][E.CI[cname]].astype(float)
            # use the 'full' model for all rows here (FLUX unseen; others seen) - diagnostic only
            dd = dict(
                auc_fakes_vs_raise_test=roc_auc_score(np.r_[np.zeros(len(Rr)), np.ones(len(F))], np.r_[s[Rr], s[F]]),
                auc_fakes_vs_vision3_test=roc_auc_score(np.r_[np.zeros(len(Rv)), np.ones(len(F))], np.r_[s[Rv], s[F]]),
                auc_vision3_vs_raise_reals=roc_auc_score(np.r_[np.zeros(len(Rr)), np.ones(len(Rv))], np.r_[s[Rr], s[Rv]]),
                median=dict(raise_test=float(np.median(s[Rr])), vision3_test=float(np.median(s[Rv])), fakes=float(np.median(s[F])),
                            raise_calib=float(np.median(s[Cr])), vision_calib=float(np.median(s[Cv])), native_test=float(np.median(s[Rn]))))
            # thresholds at 1% (or max if n<100) on calib reals of each source
            for tname, C in (("thr_raise_calib", Cr), ("thr_vision_calib", Cv)):
                t = E.metrics.threshold_for_fpr(s[C], 0.01)
                dd[tname] = dict(n_calib=len(C), fpr_raise_test=float((s[Rr] >= t).mean()), fpr_native_test=float((s[Rn] >= t).mean()),
                                 fpr_vision3_test=float((s[Rv] >= t).mean()), tpr_fakes=float((s[F] >= t).mean()))
            # threshold at 5% FPR on the source's own calib reals -> FPR on the other source
            d[cname] = dd
        diag[combo] = d
    o["diag_full_model"] = diag
    # per-device FPR of the RAISE-only probe on VISION test natives at the RAISE-calib 5% threshold
    S, _, _ = E.load_scores(arm, "std", ("F_b8", "T-aug", "raise_only"))
    s = S["full"][ci].astype(float)
    t5 = np.quantile(s[Cr], 0.95)
    dev = df.device.to_numpy()
    o["raise_only_per_device_fpr_at_raise5pct_thr_jr"] = {d_: round(float((s[Rn][dev[Rn] == d_] >= t5).mean()), 2) for d_ in sorted(set(dev[Rn]))}
    o["raise_only_raise_test_fpr_at_same_thr"] = float((s[Rr] >= t5).mean())
    o["raise_only_tpr_fakes_at_same_thr"] = float((s[F] >= t5).mean())
    out[arm] = o
(HERE / "c1_data_crosssource.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
