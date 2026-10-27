"""secondary/results/secondary.json (+ d008_guard.json, parity.json) -> secondary/tables.md.  env/.venv/bin/python -I experiments/020-attr-foundation/secondary/tables.py"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
r = json.loads((H / "results/secondary.json").read_text())
g = json.loads((H / "results/d008_guard.json").read_text())
pa = json.loads((H / "results/parity.json").read_text())
f3 = lambda x: "%.3f [%.3f, %.3f]" % tuple(x)
L = []
w = L.append
w("# 020 secondary (pre-declared DESCRIPTIVE) arms\n")
w("No labels. All CIs: cluster bootstrap (1000, seed 0) over caption families (SD) or RAID ids. Source: results/secondary.json. Frozen 020 verdicts unchanged.\n")
w("## 0. Pipeline checks\n")
w("D-008 (GPU cache vs CPU recompute, 32 new images per universe; scaled max feature diff and max |dlogit| of the 020 b8 probe, tol 1e-3):\n")
w("| universe/condition | max scaled dF | max abs dlogit (b8) | ok |\n|---|---|---|---|")
for k, v in g.items():
    w(f"| {k} | {v['max_scaled_feature_diff']:.2e} | {v['max_abs_dlogit_b8']:.2e} | {v['ok']} |")
w("\nParity (16 existing images re-extracted via this script vs the existing cache; pixel parity = raw file re-processed equals the existing processed PNG):\n")
w("| universe | n | raw->PNG pixel-identical | per-condition max scaled dF |\n|---|---|---|---|")
for k, v in pa.items():
    w(f"| {k} | {v['n']} | {sum(v['pixel_parity_raw_to_processed'])}/{v['n']} | " + ", ".join(f"{c}: {d['max_scaled']:.1e}" for c, d in v["features"].items()) + " |")
d = r["data"]
w(f"\nNew images: {d['n_new']}; {d['n_new_families']} caption families, {d['n_new_families_shared_with_old_fakes']} also used by the old 1020 fakes. {d['glide_deviation']}.\n")
w("Raw-size top (old classes): " + "; ".join(f"{k} {v}" for k, v in d["raw_size_top4"].items()) + "\n")
a = r["a_twin"]
w("## (a) Q2-near-twin: SD1.3 vs SD1.4 (chance 0.5)\n")
w(f"{a['n']} images, {a['n_families']} caption families, 5-fold StratifiedGroupKFold by family, inner-CV C (020 code). Training regime T-mix = clean + jpeg75_resize05 + social_sim; 2-AFC = same-caption pair, correct iff s(1.4)-s(1.3) is larger for the SD1.4 image.\n")
w("| regime/features | eval condition | bacc | AUROC | 2-AFC pairwise |\n|---|---|---|---|---|")
for key, v in a["metrics"].items():
    for c in ("clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"):
        m = v["by_cond"][c]
        w(f"| {key} | {c} | {f3(m['bacc'])} | {f3(m['auroc'])} | {f3(m['pw_2afc'])} |")
pn = a["perm_null_b8_tmix"]
w(f"\nLabel-swap-within-family null (b8, C=0.01, T-mix, {pn['clean']['n']} shuffles) bacc: clean mean {pn['clean']['mean']:.3f}, p95 {pn['clean']['p95']:.3f}, max {pn['clean']['max']:.3f}; jpeg75_resize05 mean {pn['jpeg75_resize05']['mean']:.3f}, p95 {pn['jpeg75_resize05']['p95']:.3f}, max {pn['jpeg75_resize05']['max']:.3f}.\n")
b = r["b_family"]
w("## (b) Q3-family\n")
nc = b["nested_choice"]
w(f"Frozen 020 protocol (T-mix, 8-way, retrained on all 1020 fakes): feature set picked by inner CV = {nc['fs']} (C={nc['C']}, inner bacc {nc['inner_bacc']:.3f}). The argmax can only be one of the 8 trained classes, so any new image is forced into one of them.\n")
cl = b["classes"]
for fsn in (nc["fs"], "b8"):
    w(f"### Predicted-class distribution, 8-way probe, feature set {fsn} (fraction of the 300 images [95% CI]; classes with >= 5% in any shown condition)\n")
    for gname in ("stable-diffusion-1-3", "stable-diffusion-1-4", "glide"):
        P = b["predicted"][gname][fsn]
        conds = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"]
        keep = [k for k in cl if any(P[c]["frac"][k][0] >= 0.05 for c in conds)]
        w(f"**{gname}**\n")
        w("| condition | " + " | ".join(keep) + " |\n|---|" + "---|" * len(keep))
        for c in conds:
            w(f"| {c} | " + " | ".join("%.2f [%.2f, %.2f]" % tuple(P[c]["frac"][k]) for k in keep) + " |")
        w("")
c11 = b["closed11"]
w("### 11-way closed-set probe (8 + SD1.3 + SD1.4 + GLIDE), T-mix, nested feature set per fold, 5-fold grouped CV (chance 1/11 = 0.091)\n")
w("| condition | bacc |\n|---|---|")
for c, v in c11["by_cond"].items():
    w(f"| {c} | {f3(v['bacc'])} |")
w("\nPer-class recall at jpeg75_resize05: " + ", ".join(f"{k} {v:.2f}" for k, v in c11["by_cond"]["jpeg75_resize05"]["recall"].items()) + "\n")
import numpy as np
C = np.array(c11["by_cond"]["jpeg75_resize05"]["confusion"])
w("Confusion rows for the new classes at jpeg75_resize05 (columns = " + ", ".join(c11["classes"]) + "):\n")
for i in (8, 9, 10):
    w(f"- {c11['classes'][i]}: {C[i].tolist()} (SD1.3+SD1.4 pooled correct {int(C[8,8]+C[8,9]+C[9,8]+C[9,9])}/600 for the two SD1.x classes)" if i == 8 else f"- {c11['classes'][i]}: {C[i].tolist()}")
pm = c11["by_cond"]["jpeg75_resize05"]["pair_matrix"]
w("\n2-AFC pairwise (jpeg75_resize05, same-caption pairs): " + ", ".join(f"{k} {v:.3f}" for k, v in pm.items() if "stable-diffusion-1" in k or "glide" in k and "dalle2" in k or k.startswith("stable-diffusion-2|stable-diffusion-1")) + "\n")
c = r["c_raid"]
w("## (c) RAID replication, n=500 ids (100 old + 400 new), 2000 fakes, 4-way (chance 0.25), T-mix = clean + jpeg(75) + jpeg(90), nested feature set\n")
w("Frozen n=100 reference (020 results.json): H-A1 RAID bacc 0.688 [0.650, 0.725] at jpeg(75), T-mix. gen_k -> generator mapping is [unverified].\n")
w("| regime | eval cond | 4-way bacc | macro recall gen_1-3 (4-way model, 'without gen_0') | 2-AFC pairwise macro | pairwise w/o gen_0 pairs | bacc old 100 ids | bacc new 400 ids |\n|---|---|---|---|---|---|---|---|")
for rg, v in c["by_regime"].items():
    for cn, m in v["by_cond"].items():
        w(f"| {rg} | {cn} | {f3(m['bacc'])} | {f3(m['bacc_without_gen0_macro'])} | {f3(m['pw_2afc_macro'])} | {f3(m['pw_2afc_without_gen0'])} | {f3(m['bacc_subsets']['old100'])} | {f3(m['bacc_subsets']['new400'])} |")
w("\nPer-class recall (gen_0..gen_3), T-mix: " + "; ".join(f"{cn}: " + ", ".join(f"{x:.2f}" for x in m["recall"].values()) for cn, m in c["by_regime"]["tmix"]["by_cond"].items()) + "\n")
w("Fresh 3-way probe on gen_1-3 only (no gen_0 rows; chance 0.333):\n")
w("| regime | eval cond | bacc | 2-AFC macro |\n|---|---|---|---|")
for rg, v in c["fresh_3way_gen123"].items():
    for cn, m in v["by_cond"].items():
        w(f"| {rg} | {cn} | {f3(m['bacc'])} | {f3(m['pw_2afc_macro'])} |")
w("\nRaw sizes of the new RAID ids: " + "; ".join(f"{k} {v}" for k, v in c["new_raw_sizes"].items()))
(H / "tables.md").write_text("\n".join(L) + "\n")
