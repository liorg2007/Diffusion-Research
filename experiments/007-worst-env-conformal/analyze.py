"""H2 (worst-environment conformal FPR) analysis. PILOT (Amendment 2 of docs/04-phase4-plan.md).

  python -I experiments/007-worst-env-conformal/analyze.py --scores experiments/007-worst-env-conformal/pilot/scores_pretrained.csv.gz
  python -I experiments/007-worst-env-conformal/analyze.py --scores experiments/005-nuisance-projection/pilot/scores_long.csv.gz

Input: long csv(.gz) with columns arm, model, condition, sha256, score (higher = more fake).
  pretrained detectors: model == "pretrained"; 005 arms (A0/A1/A2): model in {"full", "logo_<g>"}.
Outputs (next to this script, in pilot/): results_<stem>.json, tables_<stem>.md, where <stem> is the scores file stem
(without .csv / .gz). Pure numpy/pandas/scipy; seconds on CPU. Pre-registered criteria: docs/04-phase4-plan.md H2
(+ Amendment 2) and kb/ideas/worst-env-conformal-fpr.md; they are encoded verbatim in `verdict()`.
Rule used everywhere: a real/fake is FLAGGED fake iff score > threshold (strict, as in conformal prediction).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLITS = ROOT / "experiments/004-phase4-features/pilot/splits_pilot.csv"
ALPHA = 0.01
METHODS = ["M0", "M1", "M2", "M3", "M4"]
POOL2PLAT = {"vision_native": "native", "vision_whatsapp": "WA", "vision_facebook_hq": "FBH"}
PLATFORMS = ["native", "WA", "FBH", "jpeg75_resize05", "screenshot_sim"]   # held-out platforms for LOPO
N_BOOT = 2000
SEED = 0


# ----------------------------------------------------------------------------- statistics
def cp_ci(k, n, conf=0.95):
    """Clopper-Pearson two-sided interval for k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def rate(flags):
    flags = np.asarray(flags, bool)
    k, n = int(flags.sum()), int(len(flags))
    lo, hi = cp_ci(k, n)
    return dict(k=k, n=n, rate=(k / n if n else float("nan")), cp_lo=lo, cp_hi=hi)


# ----------------------------------------------------------------------------- thresholds
def thr_conformal(s, alpha):
    """Pooled split-conformal: ceil((n+1)(1-alpha))-th smallest calibration score; +inf if that index exceeds n."""
    s = np.sort(np.asarray(s, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)   # tiny tolerance against float noise in (n+1)(1-alpha)
    return float("inf") if k > n else float(s[k - 1])


def thr_worst_group(s, groups, alpha):
    """M3 (H2): max over groups of the group's empirical (1-alpha) quantile (np.quantile, method='higher')."""
    s = np.asarray(s, float)
    groups = np.asarray(groups)
    return float(max(np.quantile(s[groups == g], 1 - alpha, method="higher") for g in np.unique(groups)))


def thresholds(cal_s, cal_dev, m1, alpha=ALPHA):
    return {"M0": 0.0, "M1": float(m1), "M2": thr_conformal(cal_s, alpha),
            "M3": thr_worst_group(cal_s, cal_dev, alpha), "M4": thr_conformal(cal_s, alpha / 2)}


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, splits_csv):
        df = pd.read_csv(splits_csv, keep_default_na=False, dtype={"vidx": str})
        self.df = df
        r = df[df.label == 0]
        self.test = r[(r.split == "test") & r.pool.isin(POOL2PLAT)].copy()
        self.test["plat"] = self.test.pool.map(POOL2PLAT)
        self.test["photo"] = self.test.device + "_" + self.test.vidx
        self.calib = r[(r.split == "calib") & r.pool.isin(POOL2PLAT)].copy()
        self.fakes = df[df.label == 1].copy()
        self.devices = sorted(self.test.device.unique())
        assert len(self.devices) == 25, len(self.devices)
        assert self.calib.device.nunique() == 10 and not set(self.calib.device) & set(self.devices)


def sc(S, cond, shas):
    """scores of `shas` under `cond` from Series-dict S; raises KeyError if missing."""
    return S[cond].loc[list(shas)].to_numpy(float)


# ----------------------------------------------------------------------------- core per (arm, model)
def fpr_block(D, S):
    """LODO + LOPO FPR for all five rules for one score set S = {condition: Series indexed by sha256}."""
    T, C = D.test, D.calib
    clean_T = pd.Series(sc(S, "clean", T.sha256), index=T.sha256)
    m1 = np.quantile(sc(S, "clean", C.sha256), 1 - ALPHA, method="higher")
    out = {"M1_value": float(m1)}
    # --- LODO
    flags = {m: [] for m in METHODS}
    per_dev = {m: {} for m in METHODS}
    boot_rows = []
    for d in D.devices:
        cal, te = T[T.device != d], T[T.device == d]
        th = thresholds(clean_T.loc[cal.sha256].to_numpy(), cal.device.to_numpy(), m1)
        ts = clean_T.loc[te.sha256].to_numpy()
        for m in METHODS:
            f = ts > th[m]
            flags[m].append(f)
            per_dev[m][d] = float(f.mean())
    lodo = {}
    photo = np.concatenate([T[T.device == d].photo.to_numpy() for d in D.devices])
    for m in METHODS:
        f = np.concatenate(flags[m])
        r = rate(f)
        r["max_device_fpr"] = float(max(per_dev[m].values()))
        r["worst_device"] = max(per_dev[m], key=per_dev[m].get)
        r["cluster_boot_ci95"] = cluster_boot(f, photo)
        lodo[m] = r
    out["LODO"] = lodo
    # --- LOPO
    lopo = {}
    for P in PLATFORMS:
        if P in ("native", "WA", "FBH"):
            cal, te = T[T.plat != P], T[T.plat == P]
            cal_s, te_s = clean_T.loc[cal.sha256].to_numpy(), clean_T.loc[te.sha256].to_numpy()
        else:
            if P not in S:
                continue
            cal = T
            te = T[T.plat == "native"]
            cal_s, te_s = clean_T.loc[cal.sha256].to_numpy(), sc(S, P, te.sha256)
        th = thresholds(cal_s, cal.device.to_numpy(), m1)
        lopo[P] = {m: rate(te_s > th[m]) for m in METHODS}
        lopo[P]["thresholds"] = th
    out["LOPO"] = lopo
    return out


def cluster_boot(flags, clusters):
    """Supplementary (not used for verdicts): 95% percentile CI for the FPR, resampling photos (the same photo appears
    in native+WA+FBH, so the 3 versions are strongly dependent) with seed 0."""
    cl = pd.Series(np.asarray(flags, float)).groupby(np.asarray(clusters)).agg(["sum", "count"])
    s, c = cl["sum"].to_numpy(), cl["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    v = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def tpr_block(D, S, fakes):
    """Thresholds fit on all 25 test devices' clean reals (M1: calib devices); TPR on `fakes` rows per condition."""
    T, C = D.test, D.calib
    th = thresholds(sc(S, "clean", T.sha256), T.device.to_numpy(), np.quantile(sc(S, "clean", C.sha256), 1 - ALPHA, method="higher"))
    out = {"thresholds": th, "n_fakes": int(len(fakes))}
    for cond in ["clean", "jpeg75_resize05", "screenshot_sim"]:
        if cond not in S:
            continue
        fs = sc(S, cond, fakes.sha256)
        out[cond] = {m: float((fs > th[m]).mean()) for m in METHODS}
    return out


def ratio(t, cond):
    a, b = t[cond]["M3"], t[cond]["M2"]
    return float(a / b) if b > 0 else float("nan")


# ----------------------------------------------------------------------------- verdict (FIXED criteria)
def verdict(fp, rat_clean, rat_jpeg):
    L, P = fp["LODO"], fp["LOPO"]
    m3 = L["M3"]
    c_lodo = bool(m3["rate"] <= 0.015 and m3["cp_hi"] <= 0.02)
    c_lopo = bool(all(P[p]["M3"]["rate"] <= 0.03 for p in P))
    c_ratio = bool(rat_clean >= 0.7 and rat_jpeg >= 0.7)   # NaN -> False
    exceed = [f"LODO/{m}" for m in ["M0", "M1", "M2"] if L[m]["rate"] > 0.02]
    exceed += [f"LOPO/{p}/{m}" for p in P for m in ["M0", "M1", "M2"] if P[p][m]["rate"] > 0.02]
    c_prob = bool(exceed)
    fail_lodo = bool(m3["rate"] > 0.02)
    fail_ratio = bool(rat_clean < 0.5 or rat_jpeg < 0.5)
    if fail_lodo or fail_ratio:
        v = "FAIL"
    elif c_lodo and c_lopo and c_ratio and c_prob:
        v = "PASS"
    else:
        v = "INCONCLUSIVE"
    return dict(verdict=v, M3_LODO_ok=c_lodo, LOPO_M3_le_3pct_all=c_lopo, ratio_ge_0p7=c_ratio,
                problem_real_M0_M1_M2_gt_2pct=c_prob, exceeding=exceed, fail_by_LODO_gt_2pct=fail_lodo,
                fail_by_ratio_lt_0p5=fail_ratio, ratio_clean=rat_clean, ratio_jpeg75_resize05=rat_jpeg)


# ----------------------------------------------------------------------------- driver
def analyze_arm(D, arm, models):
    """models: {model_name: {cond: Series}}"""
    res = {"models": sorted(models)}
    logos = {m[len("logo_"):]: S for m, S in models.items() if m.startswith("logo_")}
    fpr_name = "pretrained" if "pretrained" in models else ("full" if "full" in models else None)
    if fpr_name is None:
        fpr_name = sorted(models)[0]
    res["fpr_model"] = fpr_name
    res["FPR"] = fpr_block(D, models[fpr_name])
    if logos:
        per = {g: fpr_block(D, S) for g, S in logos.items()}
        avg = {"LODO": {}, "LOPO": {}}
        for m in METHODS:
            avg["LODO"][m] = float(np.mean([per[g]["LODO"][m]["rate"] for g in per]))
            avg["LODO"][m + "_max_over_models"] = float(np.max([per[g]["LODO"][m]["rate"] for g in per]))
        for P in PLATFORMS:
            if all(P in per[g]["LOPO"] for g in per):
                avg["LOPO"][P] = {m: float(np.mean([per[g]["LOPO"][P][m]["rate"] for g in per])) for m in METHODS}
        res["FPR_logo_avg"] = avg
        res["FPR_logo_per_model"] = {g: {"LODO": {m: per[g]["LODO"][m]["rate"] for m in METHODS},
                                         "LOPO": {P: {m: per[g]["LOPO"][P][m]["rate"] for m in METHODS} for P in per[g]["LOPO"]}}
                                     for g in per}
    # TPR
    if logos:
        pg = {g: tpr_block(D, S, D.fakes[D.fakes.generator == g]) for g, S in logos.items()}
        macro = {}
        for cond in ["clean", "jpeg75_resize05", "screenshot_sim"]:
            if all(cond in pg[g] for g in pg):
                macro[cond] = {m: float(np.mean([pg[g][cond][m] for g in pg])) for m in METHODS}
        tp = {"per_generator": pg, "macro": macro, "kind": "LOGO (model logo_<g> on generator g, thresholds from logo_<g> real scores)"}
        rc = macro["clean"]["M3"] / macro["clean"]["M2"] if macro["clean"]["M2"] > 0 else float("nan")
        rj = macro["jpeg75_resize05"]["M3"] / macro["jpeg75_resize05"]["M2"] if macro["jpeg75_resize05"]["M2"] > 0 else float("nan")
        tp["ratio_per_generator"] = {g: {c: ratio(pg[g], c) for c in ["clean", "jpeg75_resize05"]} for g in pg}
    else:
        S = models[fpr_name]
        allf = tpr_block(D, S, D.fakes)
        pg = {g: tpr_block(D, S, D.fakes[D.fakes.generator == g]) for g in sorted(D.fakes.generator.unique())}
        macro = {c: {m: float(np.mean([pg[g][c][m] for g in pg])) for m in METHODS} for c in ["clean", "jpeg75_resize05", "screenshot_sim"] if c in allf}
        tp = {"pooled": allf, "per_generator": pg, "macro": macro, "kind": "pooled over all 720 fakes (pretrained detector)"}
        rc, rj = ratio(allf, "clean"), ratio(allf, "jpeg75_resize05")
        tp["ratio_per_generator"] = {g: {c: ratio(pg[g], c) for c in ["clean", "jpeg75_resize05"]} for g in pg}
    tp["ratio_M3_over_M2"] = {"clean": float(rc), "jpeg75_resize05": float(rj)}
    res["TPR"] = tp
    res["verdict"] = verdict(res["FPR"], rc, rj)
    if logos:   # robustness line: same criteria with LOGO-averaged FPR rates (CP bound of the average not defined)
        a = res["FPR_logo_avg"]
        res["verdict"]["logo_avg_check"] = dict(
            M3_LODO_avg=a["LODO"]["M3"], M3_LODO_max_over_models=a["LODO"]["M3_max_over_models"],
            LOPO_M3_avg={P: a["LOPO"][P]["M3"] for P in a["LOPO"]},
            M3_LODO_avg_le_1p5=bool(a["LODO"]["M3"] <= 0.015),
            LOPO_M3_avg_le_3=bool(all(a["LOPO"][P]["M3"] <= 0.03 for P in a["LOPO"])))
    if arm == "A0":
        res["primary"] = True
    return res


def pct(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def tables(R, stem):
    L = [f"# PILOT H2 tables ({stem})", "",
         "PILOT (Amendment 2): partial data (VISION + 6 Synthbuster generators), not a Phase 4 verdict. alpha = 1%. "
         "Flag fake iff score > threshold. LODO/LOPO details in notes.md.", ""]
    for arm, r in R["arms"].items():
        F, T = r["FPR"], r["TPR"]
        L += [f"## {arm}{' (pre-registered primary)' if r.get('primary') else ''}  [FPR model: {r['fpr_model']}]", ""]
        L += ["### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)", "",
              "| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            x = F["LODO"][m]
            L.append(f"| {m} | {pct(x['rate'])} | {x['k']}/{x['n']} | [{pct(x['cp_lo'])}, {pct(x['cp_hi'])}] | "
                     f"{pct(x['max_device_fpr'])} ({x['worst_device']}) | [{pct(x['cluster_boot_ci95'][0])}, {pct(x['cluster_boot_ci95'][1])}] |")
        L += ["", "### LOPO FPR (held-out platform; CP 95% CI)", "",
              "| platform (n) | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
        for P, x in F["LOPO"].items():
            L.append(f"| {P} ({x['M0']['n']}) | " + " | ".join(f"{pct(x[m]['rate'])} [{pct(x[m]['cp_lo'])}, {pct(x[m]['cp_hi'])}]" for m in METHODS) + " |")
        L += ["", f"Thresholds on all 25 test devices (clean reals; M1 from the 10 calib devices): "
              + ", ".join(f"{m}={T['pooled']['thresholds'][m]:.3f}" for m in METHODS) if "pooled" in T else
              "Thresholds: per logo_<g> model (see results json).", ""]
        if "FPR_logo_avg" in r:
            a = r["FPR_logo_avg"]
            L += ["### LOGO-averaged FPR (mean over logo_<g> models)", "",
                  "| split | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS),
                  "| LODO | " + " | ".join(pct(a["LODO"][m]) for m in METHODS) + " |"]
            for P, x in a["LOPO"].items():
                L.append(f"| LOPO {P} | " + " | ".join(pct(x[m]) for m in METHODS) + " |")
            L.append("")
        L += [f"### TPR ({T['kind']}); thresholds fit on all 25 test devices' clean reals", "",
              "| condition | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
        tpr_src = T["pooled"] if "pooled" in T else T["macro"]
        lab = "pooled" if "pooled" in T else "macro mean over g"
        for c in ["clean", "jpeg75_resize05", "screenshot_sim"]:
            if c in tpr_src:
                L.append(f"| {c} ({lab}) | " + " | ".join(pct(tpr_src[c][m], 1) for m in METHODS) + " |")
        rr = T["ratio_M3_over_M2"]
        L += ["", f"TPR ratio M3/M2: clean {rr['clean']:.3f}, jpeg75_resize05 {rr['jpeg75_resize05']:.3f}", "",
              "Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):", "", "| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |", "|---|---|---|---|---|---|---|"]
        for g, x in T["per_generator"].items():
            rg = T["ratio_per_generator"][g]
            L.append(f"| {g} | {pct(x['clean']['M2'], 1)} | {pct(x['clean']['M3'], 1)} | {rg['clean']:.2f} | "
                     f"{pct(x['jpeg75_resize05']['M2'], 1)} | {pct(x['jpeg75_resize05']['M3'], 1)} | {rg['jpeg75_resize05']:.2f} |")
        v = r["verdict"]
        L += ["", f"### PILOT verdict: **{v['verdict']}**", "",
              f"- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: {v['M3_LODO_ok']}",
              f"- M3 LOPO FPR <= 3% on every held-out platform: {v['LOPO_M3_le_3pct_all']}",
              f"- TPR ratio >= 0.7 (clean {v['ratio_clean']:.3f}, jpeg75_resize05 {v['ratio_jpeg75_resize05']:.3f}): {v['ratio_ge_0p7']}",
              f"- some M0/M1/M2 FPR > 2% on a split: {v['problem_real_M0_M1_M2_gt_2pct']} ({', '.join(v['exceeding']) or 'none'})",
              f"- fail triggers: M3 LODO > 2%: {v['fail_by_LODO_gt_2pct']}; ratio < 0.5: {v['fail_by_ratio_lt_0p5']}", ""]
        if "logo_avg_check" in v:
            L += [f"- LOGO-averaged check: {json.dumps(v['logo_avg_check'])}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--splits", default=str(SPLITS))
    ap.add_argument("--out-dir", default=str(HERE / "pilot"))
    a = ap.parse_args()
    D = Data(a.splits)
    p = Path(a.scores)
    stem = p.name
    for suf in (".gz", ".csv"):
        stem = stem[:-len(suf)] if stem.endswith(suf) else stem
    long = pd.read_csv(p, dtype={"sha256": str})
    assert {"arm", "model", "condition", "sha256", "score"} <= set(long.columns), long.columns
    assert long.score.notna().all() and np.isfinite(long.score).all(), "non-finite scores"
    R = {"PILOT": True, "scores_file": str(p), "alpha": ALPHA, "seed": SEED, "arms": {}}
    for arm, g in long.groupby("arm", sort=True):
        models = {}
        for model, gm in g.groupby("model"):
            S = {}
            for cond, gc in gm.groupby("condition"):
                s = gc.set_index("sha256").score
                assert s.index.is_unique, f"duplicate rows {arm}/{model}/{cond}"
                S[cond] = s
            need = set(D.test.sha256) | set(D.calib.sha256)
            if "clean" not in S or not need <= set(S["clean"].index):
                print(f"skip {arm}/{model}: clean reals incomplete"); continue
            models[model] = S
        if not models:
            continue
        R["arms"][str(arm)] = analyze_arm(D, str(arm), models)
        print(f"{arm}: {R['arms'][str(arm)]['verdict']['verdict']}")
    # sanity: CommFor-style M0 FPR on test natives (and WA/FBH) for every arm/FPR model
    sanity = {}
    for arm, r in R["arms"].items():
        S = {}
        sub = long[(long.arm == arm) & (long.model == r["fpr_model"]) & (long.condition == "clean")].set_index("sha256").score
        for plat in ["native", "WA", "FBH"]:
            sh = D.test[D.test.plat == plat].sha256
            sanity[f"{arm}/{plat}"] = rate(sub.loc[sh].to_numpy() > 0)
    R["sanity_M0_clean_FPR_test_reals_by_platform"] = sanity
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True, parents=True)
    (out / f"results_{stem}.json").write_text(json.dumps(R, indent=1))
    (out / f"tables_{stem}.md").write_text(tables(R, stem))
    print("wrote", out / f"results_{stem}.json", out / f"tables_{stem}.md")


if __name__ == "__main__":
    main()
