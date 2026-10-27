"""tables.md writer for 023-attr-detector-emb (reads the results dict built by run.py stage_report)."""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def fc(c, d=3):
    return f"{f(c[0], d)} [{f(c[1], d)}, {f(c[2], d)}]"


def write_tables(R, A):
    L = []
    P = R["primary"]
    conds = A.conds
    det = list(R["config"]["detectors"])
    dino = R["config"]["ha1_probe"]["dino_fs"]
    fs_all = dino + ["nested_dino"] + det
    lab = R["labels"]["H-A5"]
    L.append("# 023-attr-detector-emb: H-A5 tables (all numbers from results/results.json)\n")
    L.append(f"**H-A5 label: {lab['label']}**. Best detector embedding: {lab.get('best_detector')} (bacc {f(lab.get('best_bacc'))} at {P}, T-mix). "
             f"H-A1 best fixed layer: {lab.get('a1_best_layer')} (bacc {f(lab.get('a1_best_layer_bacc', lab.get('a1_best_bacc')))}); H-A1 nested-choice bacc {f(lab.get('a1_nested_bacc'))}. "
             f"Rule: PASS if >= best layer + {R['config']['label_rule']['margin']} and paired CI lower bound > 0; KILL if <= best layer; else INCONCLUSIVE. Gates all ok: {R['labels'].get('all_gates_ok')}.\n")
    L.append(f"n = {R['n']} fakes, {R['n_clusters']} caption families (cluster bootstrap, {R['config']['boot']['n_boot']} reps, 95% percentile CI). Chance bacc = {f(R['chance'])}, pairwise/AUROC chance = 0.5.\n")
    for regime, title in (("tmix", "T-mix training (headline)"), ("clean", "clean-train transfer (descriptive)")):
        L.append(f"## Balanced accuracy, 8-way, {title}\n")
        L.append("| embedding | " + " | ".join(conds) + " |")
        L.append("|---|" + "---|" * len(conds))
        for fs in fs_all:
            key = f"{regime}|{fs}"
            if key not in R["fs"]:
                continue
            S = R["fs"][key]["summary"]
            L.append(f"| {fs} | " + " | ".join(fc(S[c]["bacc"]) for c in conds) + " |")
        maj = R["fs"][f"tmix|{dino[0]}"]["summary"][P]["majority_rate"]
        L.append(f"| chance / majority rate | " + " | ".join(f"{f(R['chance'])} / {f(maj)}" for _ in conds) + " |\n")
    L.append(f"## Secondary metrics at {P} (T-mix)\n")
    L.append("| embedding | macro-F1 | bacc w/o firefly | pairwise 2-AFC macro | pairwise pooled | pairwise strict macro | n pairs |")
    L.append("|---|---|---|---|---|---|---|")
    for fs in fs_all:
        S = R["fs"][f"tmix|{fs}"]["summary"][P]
        L.append(f"| {fs} | {fc(S['mf1'])} | {fc(S['bacc_drop'])} | {fc(S['pw']['macro'])} | {fc(S['pw']['pooled'])} | {fc(S['pw_strict']['macro'])} | {S['pw']['n_pairs']} |")
    L.append("\nChance for pairwise = 0.5 (2-AFC and strict reading: strict chance 0.25 for independent coin flips).\n")
    pv = R["paired_vs_a1_best_layer"]
    L.append(f"## Paired comparison with H-A1 best fixed layer ({pv['a1_best_layer']}, bacc {f(pv['a1_best_bacc'])}) at {P}, T-mix\n")
    L.append("| detector | bacc | diff = detector - H-A1 best layer [95% paired CI] | one-sided p | Holm p |")
    L.append("|---|---|---|---|---|")
    for d in det:
        b = R["fs"][f"tmix|{d}"]["summary"][P]["bacc"][0]
        L.append(f"| {d} | {f(b)} | {fc(pv['det_minus_a1'][d])} | {f(pv['p_one_sided'][d], 4)} | {f(pv['p_holm'][d], 4)} |")
    L.append("\n")
    L.append("### Same diff for every condition and every DINOv2 layer (descriptive, no label)\n")
    L.append("| detector - layer | " + " | ".join(conds) + " |")
    L.append("|---|" + "---|" * len(conds))
    for d in det:
        for fs in dino:
            L.append(f"| {d} - {fs} | " + " | ".join(fc(R["paired_all_descriptive"][f"{c}|{d}-{fs}"]) for c in conds) + " |")
    L.append(f"\n## Per-class recall at {P} (T-mix), with training-set-overlap flags\n")
    L.append("| generator | " + " | ".join(fs_all) + " |")
    L.append("|---|" + "---|" * len(fs_all))
    for g in R["classes"]:
        L.append(f"| {g} | " + " | ".join(f(R["fs"][f"tmix|{fs}"]["summary"][P]["recall"][g]) for fs in fs_all) + " |")
    L.append("\nOverlap flags (family-level, from kb/papers; [unverified] where the exact checkpoint lists are not public to us):\n")
    L.append("| generator | " + " | ".join(det) + " |")
    L.append("|---|" + "---|" * len(det))
    of = R["overlap_flags"]
    for g in R["classes"]:
        cells = []
        for d in det:
            v = of[d].get(g)
            if v is None:
                v = of[d].get("others", "none known") if g not in ("stable-diffusion-2", "stable-diffusion-xl") else of[d].get(g, "none known")
            cells.append(v)
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    L.append("\nNotes: " + "; ".join(f"{d}: {of[d]['note']}" for d in det if "note" in of[d]) + "\n")
    L.append("## Open set (leave-one-generator-out), T-mix\n")
    for sc, name in (("msp", "-max softmax"), ("maha", "Mahalanobis (Ledoit-Wolf)")):
        L.append(f"### AUROC, novelty = {name}, {P}\n")
        L.append("| embedding | " + " | ".join(R["classes"]) + " | mean | min | usable (mean>=0.80, min>=0.65) |")
        L.append("|---|" + "---|" * (len(R["classes"]) + 3))
        for fs in dino + det:
            o = R["logo"][f"tmix|{fs}"][sc][P]
            vals = [o[g][0] for g in R["classes"]]
            u = R["open_set"][fs][sc]
            L.append(f"| {fs} | " + " | ".join(fc(o[g], 2) for g in R["classes"]) + f" | {f(np.mean(vals))} | {f(np.min(vals))} | {u['usable']} |")
        L.append("")
    L.append("## Held-out launderings (never in training), bacc T-mix\n")
    L.append("| embedding | screenshot_sim | mavali_medium |")
    L.append("|---|---|---|")
    for fs in fs_all:
        S = R["fs"][f"tmix|{fs}"]["summary"]
        L.append(f"| {fs} | {fc(S['screenshot_sim']['bacc'])} | {fc(S['mavali_medium']['bacc'])} |")
    if R.get("baselines_020"):
        b = R["baselines_020"]
        L.append("\n## Baselines from 020 (same rows and folds)\n")
        L.append(f"M0 raw metadata bacc {fc(b['M0_raw'])}; M1 texture bacc (T-mix) {fc(b['M1'])} at {P}. M0 on preprocessed tensors (must be near chance): {json.dumps(b.get('M0_tensor'))}.\n")
    L.append("\n## Gates\n")
    for k, g in R["gates"].items():
        L.append(f"- {'OK  ' if g['ok'] else 'FAIL'} {k}: " + json.dumps({a: v for a, v in g.items() if a != 'ok'}, default=str)[:400])
    c = R.get("consistency_020", {})
    L.append(f"\n020 results available for cross-check: {c.get('available')}; max |bacc diff| vs 020 results.json (primary, T-mix): {c.get('max_abs_diff')}; fold identity: {R.get('fold_identity_with_020')}")
    L.append("\n## D-008 (GPU vs CPU, 32 images x 5 conditions)\n")
    gd = json.loads((HERE / "results" / "d008_guard.json").read_text())
    L.append("| detector | worst scaled emb diff | worst abs emb diff | worst |dlogit| | all ok |")
    L.append("|---|---|---|---|---|")
    for d in det:
        v = gd[d].values()
        L.append(f"| {d} | {max(x['max_scaled_emb_diff'] for x in v):.2e} | {max(x['max_abs_emb_diff'] for x in v):.2e} | {max(x['max_abs_dlogit'] for x in v):.2e} | {all(x['ok'] for x in v)} |")
    L.append("\n## Runtime\n")
    for d in det:
        t = R["timings"][d]
        L.append(f"- {d}: extraction {sum(v['seconds'] for v in t.values()):.0f} s over {len(t)} conditions")
    (HERE / "results" / "tables.md").write_text("\n".join(L) + "\n")
