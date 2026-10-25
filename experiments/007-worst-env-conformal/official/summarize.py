"""Compact summary of official H2-v2 results for one arm: python -I summarize.py orig|fov512 -> summary_<arm>.md"""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent; arm = sys.argv[1]
def load(tag):
    p = H / f"{arm}_{tag}" / f"results_v2_scores_long_{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None
pc = lambda x: f"{100*x:.2f}%"
R = load("primary_platform_min100"); S = load("sens_platform_brand_min50")
gens = sorted(p.name[len(f"{arm}_primary_drop_"):] for p in H.glob(f"{arm}_primary_drop_*") if p.is_dir())
L = [f"# Summary OFFICIAL H2-v2, arm {arm}", ""]
def block(R, gk, title):
    L.append(f"## {title}"); 
    L.append("| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    order = ["B0", "B0_twinfree"] + sorted(a for a in R["arms"] if not a.startswith("B0"))
    for a in order:
        if a not in R["arms"]: continue
        r = R["arms"][a][gk]; ru = r["rules"]
        wp = lambda k: max(ru[k]["fpr_by_platform"].items(), key=lambda kv: kv[1])
        f = lambda k: f"{pc(ru[k]['fpr_pooled'])}"
        wpl = lambda k: f"{pc(wp(k)[1])} ({wp(k)[0]})"
        L.append(f"| {a} | {f('M1')} / {f('pooled@0.01')} / {f('wg@0.01')} | {wpl('M1')} / {wpl('pooled@0.01')} / {wpl('wg@0.01')} | "
                 f"{pc(ru['pooled@0.01']['fpr_worst_group'])} / {pc(ru['wg@0.01']['fpr_worst_group'])} | "
                 f"{pc(ru['pooled@0.01']['tpr']['clean']['value'])} -> {pc(ru['wg@0.01']['tpr']['clean']['value'])} | "
                 f"{pc(ru['pooled@0.01']['tpr']['jpeg75_resize05']['value'])} -> {pc(ru['wg@0.01']['tpr']['jpeg75_resize05']['value'])} | "
                 f"{r['clipped']['wg_group_rate']['0.01']:.2f} | {r['verdict']['verdict']} ({r['verdict']['n_alphas_ci_upper_lt0']} CI<0, {r['verdict']['n_alphas_diff_ge0']} diff>=0) | {r['verdict']['problem_real_M1_or_M2_gt_2pct']} |")
    L.append("")
if R: block(R, "platform", "PRIMARY: platform-only groups, min-group 100")
if S:
    for gk in S["arms"]["B0"]: block(S, gk, f"SENSITIVITY: grouping {gk}, min-group 50")
L.append("## Verdict robustness (Amendment 13): primary verdict with Firefly out / each generator dropped"); 
head = ["full"] + gens
L.append("| detector | " + " | ".join(head) + " |"); L.append("|---|" + "---|" * len(head))
Rs = {"full": R}; Rs.update({g: load(f"primary_drop_{g}") for g in gens})
for a in R["arms"]:
    L.append(f"| {a} | " + " | ".join(Rs[h]["arms"][a]["platform"]["verdict"]["verdict"].replace("NOT_SUPPORTED","NS").replace("INCONCLUSIVE","INC").replace("SUPPORTED","SUP") if Rs[h] and a in Rs[h]["arms"] else "-" for h in head) + " |")
L.append("")
if S and load("sens_platform_brand_min50_drop_firefly"):
    F = load("sens_platform_brand_min50_drop_firefly")
    L.append("Sensitivity (brand-50) with Firefly out: " + "; ".join(f"{a}: {F['arms'][a]['platform_brand']['verdict']['verdict']}" for a in F["arms"]))
grp = R["grouping_info"]["platform"]["table"]; L.append("Primary groups: " + "; ".join(f"{t['group']} n={t['n_reals']}" for t in grp))
if S: L.append("Sensitivity groups: " + "; ".join(f"{t['group']} n={t['n_reals']}" for t in S["grouping_info"]["platform_brand"]["table"]))
(H / f"summary_{arm}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
