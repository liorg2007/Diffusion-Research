"""E9 critic c3: (a) A-proj vs E7 paired AUROC gain with the cluster-bootstrap CI (same Plan replicates; row 0 = point);
(b) A-proj ranking + ANCHOR threshold (label-agnostic, deployable: 1% FPR of the anchors' own A-proj scores, clean = reading A, JR = reading B);
(c) A-proj full-model macro (c5's convention) next to the LOGO macro;
(d) scan of every label-agnostic variant in results.json (E7, A-thr, A-off-all, A-proj) for the DISCOVERY bar (AUROC >= 0.90 and FPR <= 5% on all 3 sources)."""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import numpy as np
import run as R

out = {"gain": {}, "proj_anchor_thr": {}, "proj_full_macro": {}, "scan": []}
for arm in ("fov512",):
    for S in R.SOURCES:
        e7 = pickle.load(open(R.eval_path(arm, "F_b8", S, 0, "E7"), "rb"))["boot_auc"]
        for n in (50, 100):
            ap = pickle.load(open(R.eval_path(arm, "F_b8", S, n, "A-proj"), "rb"))["boot_auc"]
            d = ap - e7
            out["gain"][f"{S}|n={n}"] = dict(e7=float(e7[0]), aproj=float(ap[0]), aproj_ci=[float(np.percentile(ap[1:], 2.5)), float(np.percentile(ap[1:], 97.5))],
                                             gain=float(d[0]), gain_ci=[float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))],
                                             P_aproj_ge_0_90=float((ap[1:] >= 0.90).mean()))
ctx = R.get_ctx("fov512"); D = ctx.D
for S in R.SOURCES:
    rr = ctx.rr[S]; f = ctx.full
    for n in (50, 100):
        fa, fb, ta, full, logo = [], [], [], [], []
        for dr in range(R.ND):
            sc = np.load(R.proj_path("fov512", "F_b8", S, n, dr))["sc"].astype(np.float64)
            anc = ctx.anchor_order(S, dr)[:n]
            tA = R.thr_fpr(sc[f, R.CI_CLEAN, anc]); tB = R.thr_fpr(sc[f, R.CI_JR, anc])
            fa.append((sc[f, R.CI_JR, rr] >= tA).mean()); fb.append((sc[f, R.CI_JR, rr] >= tB).mean())
            ta.append(np.mean([(sc[f, R.CI_JR, ctx.ff[g]] >= tA).mean() for g in ctx.gens]))
            full.append(np.mean([R.auroc_point(sc[f, R.CI_JR, rr], sc[f, R.CI_JR, ctx.ff[g]]) for g in ctx.gens]))
            logo.append(np.mean([R.auroc_point(sc[ctx.mi[g], R.CI_JR, rr], sc[ctx.mi[g], R.CI_JR, ctx.ff[g]]) for g in ctx.gens]))
        out["proj_anchor_thr"][f"{S}|n={n}"] = dict(fpr_A=float(np.mean(fa)), fpr_A_p97_5=float(np.percentile(fa, 97.5)), fpr_B=float(np.mean(fb)), tpr_A=float(np.mean(ta)))
        out["proj_full_macro"][f"{S}|n={n}"] = dict(full=float(np.mean(full)), logo=float(np.mean(logo)))
    base, _ = R.base_scores(ctx, "F_b8", S)
    out["proj_full_macro"][f"{S}|E7"] = dict(full=float(np.mean([R.auroc_point(base[f, R.CI_JR, rr], base[f, R.CI_JR, ctx.ff[g]]) for g in ctx.gens])))
res = json.load(open(R.RES / "results.json"))["results"]
for arm in res:
    for feat in res[arm]:
        for m in ("E7", "A-thr", "A-off-all", "A-proj"):
            for n in res[arm][feat]["VISION"][m]:
                for cond in R.CONDS:
                    try:
                        per = {S: res[arm][feat][S][m][n][cond] for S in R.SOURCES}
                    except KeyError:
                        continue
                    a = {S: per[S]["auroc"][0] if isinstance(per[S]["auroc"], list) else per[S]["auroc"] for S in per}
                    fr = {S: per[S]["fpr_A"][0] if isinstance(per[S]["fpr_A"], list) else per[S]["fpr_A"] for S in per}
                    out["scan"].append(dict(arm=arm, feat=feat, method=m, n=n, cond=cond, min_auroc=min(a.values()), max_fprA=max(fr.values()),
                                            meets=bool(min(a.values()) >= 0.90 and max(fr.values()) <= 0.05)))
sc_ = out["scan"]
out["scan_summary"] = dict(n_variants=len(sc_), n_meet=sum(r["meets"] for r in sc_), n_auroc_ok=sum(r["min_auroc"] >= 0.90 for r in sc_),
                           meeting=[r for r in sc_ if r["meets"]], best_jr=sorted([r for r in sc_ if r["cond"] == R.JR], key=lambda r: -r["min_auroc"])[:6])
for k in ("gain", "proj_anchor_thr", "proj_full_macro", "scan_summary"):
    print(k, json.dumps(out[k], indent=0))
json.dump(out, open(Path(__file__).with_name("c3_aproj.json"), "w"), indent=1)
