"""Build findings.json for the demo's Findings / Phone-fingerprints tabs from the project's result files (parsed, not retyped).

    env/.venv/bin/python -I poc/fingerprint-demo/build_findings.py
Every block carries a `source` path; numbers are parsed from those files. Plain-language sentences are authored here.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = C.ROOT
F020 = "experiments/020-attr-foundation/results/tables.md"
F020S = "experiments/020-attr-foundation/secondary/tables.md"
F019 = "experiments/019-source-diversity/results/tables.md"
F021 = "experiments/021-attr-residual/results/tables.md"
F021C = "experiments/021-attr-residual/results/prnu_scores.csv"
F023 = "experiments/023-attr-detector-emb/results/tables.md"
D10 = "docs/10-attribution-report.md"
D03 = "docs/03-final-report.md"
COND5 = ["clean", "jpeg75_resize05", "social_sim", "screenshot_sim", "mavali_medium"]

PLAIN = {
    "H-A1": "A linear read-out of DINOv2's middle layers names which of 8 known generators made an image, even after JPEG-75 plus half-size resizing.",
    "H-A1 side label": "But it cannot reliably say 'none of these': its confidence separates unknown generators only weakly.",
    "H-A2": "Sensor noise (PRNU) identifies which of 14 same-size 2010s phones took a photo; for AI generators the same trick is no better than simple texture statistics.",
    "H-A3": "Which platform a photo went through (native, WhatsApp, Facebook) lives in file headers, not in laundered pixels.",
    "H-A4": "One shared space can name known sources, but does not notice unknown ones (novelty at chance).",
    "H-A5": "Detectors trained for real-vs-fake keep less generator identity in their last layer than a generic backbone's middle layers. Their own middle layers do as well or better (026 R1).",
    "H-A6": "The VAE-decoder signature is strong on untouched files and gone after one JPEG plus resize.",
}


def txt(rel):
    return (R / rel).read_text()


def tables(rel):
    return C.md_tables(R / rel)


def rowmap(t):
    return {r[0]: r for r in t["rows"]}


def scorecard():
    t = C.find_table(tables(D10), "Results per hypothesis")
    out = []
    for r in t["rows"]:
        out.append(dict(id=r[0], method=r[1], label=r[2], counts=r[3], key_number=r[4], critic=r[5], plain=PLAIN.get(r[0], "")))
    t3 = C.find_table(tables(D03), "Summary of labels")
    det = [dict(experiment=r[0], kind=r[1], label=r[2], critic=r[3]) for r in t3["rows"]]
    return dict(attribution=out, detection=det, source=D10 + " section 4; " + D03 + " section 3.3")


def laundering():
    t = C.find_table(tables(F020), "bacc per condition", header0="method / regime", nth=0)
    rm = rowmap(t)
    series = {}
    for key, name in [("nested / tmix", "Probe trained on laundered copies (T-mix)"), ("nested / clean", "Probe trained on clean images only"),
                      ("M1 / tmix", "Texture statistics baseline (M1)"), ("M0 raw metadata", "File metadata only (M0, raw files)")]:
        vals = [C.num_ci(c) for c in rm[key][1:6]]
        series[key] = dict(name=name, values=[v[0] for v in vals], lo=[v[1] for v in vals], hi=[v[2] for v in vals])
    return dict(conditions=COND5, series=series, chance=0.125, source=F020 + " ('bacc per condition')",
                sentence="Trained on laundered copies, the 8-way read-out stays at 0.81-0.91 under the launderings tested, but those are close to its training mix (a clean-only probe already gets 0.875 on screenshot_sim, 026 critique). Genuinely new edits such as mild blur or sharpening drop it to about 0.52 (exploratory). Trained on clean images only, it falls to 0.54 after JPEG-75 plus half-size (chance 0.125). File metadata alone already gives 0.69, which is why the probe only sees decoded pixels.")


def layers():
    t = C.find_table(tables(F020), "bacc per condition", header0="method / regime", nth=0)
    rm = rowmap(t)
    out = {}
    for reg in ("tmix", "clean"):
        out[reg] = {b: C.num_ci(rm[f"{b} / {reg}"][2]) for b in ("b4", "b8", "b12", "cat")}
    m = re.search(r"Layer contrast b8 - b12 bacc at jpeg75_resize05: T-mix ([^;]+); clean-train (.+)", txt(F020))
    return dict(primary="jpeg75_resize05", values={r: {b: dict(v=x[0], lo=x[1], hi=x[2]) for b, x in d.items()} for r, d in out.items()},
                contrast_b8_minus_b12=dict(tmix=m.group(1).strip(), clean=m.group(2).strip()), chance=0.125,
                source=F020 + " ('bacc per condition', 'Layer contrast')",
                sentence="The middle layers (block 4/8) carry the generator's rendering style; the last, most 'semantic' layer (block 12) carries much less. The b8 minus b12 gap is +0.254.")


def detector_emb():
    t = C.find_table(tables(F023), "T-mix training (headline)")
    rows = {r[0]: [C.num_ci(c)[0] for c in r[1:6]] for r in t["rows"] if r[0] in ("cat", "nested_dino", "commfor", "bfree", "rine", "clipdet", "b4", "b8", "b12")}
    names = {"cat": "DINOv2 b4+b8+b12", "nested_dino": "DINOv2 nested choice", "commfor": "Community Forensics detector", "bfree": "B-Free detector",
             "rine": "RINE detector", "clipdet": "CLIP-based detector", "b4": "DINOv2 b4", "b8": "DINOv2 b8", "b12": "DINOv2 b12 (last)"}
    return dict(conditions=COND5, rows=[dict(key=k, name=names[k], values=v) for k, v in rows.items()], chance=0.125,
                source=F023 + " ('Balanced accuracy, 8-way, T-mix training (headline)')",
                sentence="Detectors trained to say real-vs-fake keep less generator identity in their final embedding: the best one (0.740 at jpeg75_resize05) is 0.118 below DINOv2's own layers; RINE's probe collapsed. This compares detectors at their LAST layer only: at a middle layer Community Forensics reaches 0.883 and beats DINOv2-base (0.856; 026 R1), so it does not show generic features are better.")


def learning_curve():
    ts = tables(F019)
    mean = [t for t in ts if t["header"][0] == "k" and t["header"][1] == "AUROC"][0]
    ks = []
    for r in mean["rows"]:
        a, f = C.num_ci(r[1]), C.num_ci(r[2])
        ks.append(dict(k=int(r[0]), auroc=a[0], auroc_lo=a[1], auroc_hi=a[2], fpr=f[0], fpr_lo=f[1], fpr_hi=f[2]))
    per = [t for t in ts if t["header"][0] == "held-out" and t["header"][1] == "k=1 AUROC"][0]
    srcs = {}
    for r in per["rows"]:
        srcs[r[0]] = dict(auroc=[float(x) for x in r[1:6]], fpr=[float(x) for x in r[6:11]])
    return dict(mean=ks, per_source=srcs, source=F019 + " ('Learning curve', mean over held-out sources and per source; jpeg75_resize05, F_b8, FPR reading A)",
                sentence="Adding more kinds of real photos to training cuts the false-positive rate on an unseen real source from 40% to 7% on average, but AUROC only moves from 0.69 to 0.77: diversity mostly recalibrates the threshold.")


def unseen_fpr():
    t = C.find_table(tables(F019), "Held-out source at k = 5 (train")
    out = []
    for r in t["rows"]:
        if r[1] != "F_b8":
            continue
        m = re.match(r"\s*([\d.]+) \((\d+)/(\d+)\) \[([\d.]+), ([\d.]+)\]", r[3])
        a = C.num_ci(r[2])
        out.append(dict(source=r[0], auroc=a[0], fpr=float(m.group(1)), k=int(m.group(2)), n=int(m.group(3)), cp_lo=float(m.group(4)), cp_hi=float(m.group(5))))
    e7 = [t for t in tables(F019) if t["header"][0] == "held-out" and t["header"][1] == "condition"][0]
    e7rows = {r[0]: dict(fpr_k5=r[5], fpr_e7=r[6]) for r in e7["rows"] if r[1] == "jpeg75_resize05"}
    return dict(sources=out, e7_vs_e10=e7rows, source=F019 + " ('Held-out source at k = 5', F_b8, reading A; 'Key secondary')",
                sentence="On a real-photo source the detector never saw, its false-positive rate at the 1% threshold ranges from 0.5% (HDR+ phone finals) to 17% (Unsplash). The 'FPR <= 1% on fresh phone photos' target is untested on modern phones.")


def open_set():
    t = [x for x in tables(F020) if x["header"][0] == "held-out generator"][0]
    rows = []
    for r in t["rows"]:
        rows.append(dict(generator=r[0], msp=C.num_ci(r[1])[0], msp_lo=C.num_ci(r[1])[1], msp_hi=C.num_ci(r[1])[2], maha=C.num_ci(r[2])[0]))
    return dict(rows=rows, mean_msp=float(np.mean([r["msp"] for r in rows])), bar=0.80, source=F020 + " ('Open set: leave-one-generator-out AUROC', T-mix, jpeg75_resize05)",
                sentence="Asked 'is this from a generator I do not know?', the probe's confidence scores 0.70-0.85 AUROC per generator (mean 0.77), below the pre-set 0.80 bar. Unknown generators are silently labelled as known ones.")


def near_twins():
    t = C.find_table(tables(F020S), "Q2-near-twin")
    rows = [r for r in t["rows"] if r[0] == "tmix/b8"]
    null = re.search(r"Label-swap-within-family null \(b8[^)]*\) bacc: clean mean ([\d.]+), p95 ([\d.]+), max ([\d.]+); jpeg75_resize05 mean ([\d.]+), p95 ([\d.]+), max ([\d.]+)", txt(F020S))
    return dict(rows=[dict(condition=r[1], bacc=C.num_ci(r[2])[0], lo=C.num_ci(r[2])[1], hi=C.num_ci(r[2])[2]) for r in rows], chance=0.5,
                null_p95=dict(clean=float(null.group(2)), jpeg75_resize05=float(null.group(5))),
                source=F020S + " ('(a) Q2-near-twin', tmix/b8; label-swap null)",
                sentence="Stable Diffusion 1.3 and 1.4 render alike: in every condition the probe tells them apart no better than a coin flip (0.49-0.55, inside the shuffled-label null).")


def prnu():
    d = pd.read_csv(R / F021C)
    order = ["D02", "D10", "D05", "D14", "D18", "D06", "D15", "D29", "D34", "D04", "D11", "D19", "D21", "D33"]
    groups = [["iPhone 4s", ["D02", "D10"]], ["iPhone 5c", ["D05", "D14", "D18"]], ["iPhone 6", ["D06", "D15"]], ["iPhone 5", ["D29", "D34"]],
              ["other single-device models", ["D04", "D11", "D19", "D21", "D33"]]]
    arms = [("nat", "Native photos", "native fingerprint vs native photo"), ("wa", "WhatsApp copies", "WhatsApp fingerprint vs WhatsApp photo"),
            ("fbh", "Facebook copies", "Facebook (HQ) fingerprint vs Facebook photo"),
            ("nat2wa", "Native fingerprint on WhatsApp", "native fingerprint, resized, vs WhatsApp photo")]
    out = {}
    for arm, nice, desc in arms:
        a = d[d.arm == arm]
        med = np.zeros((14, 14))
        mx = np.zeros((14, 14))
        share = np.zeros((14, 14))
        for i, t_ in enumerate(order):
            for j, c_ in enumerate(order):
                v = a[(a.dev == t_) & (a.cand == c_)].pce.to_numpy()
                med[i, j], mx[i, j], share[i, j] = float(np.median(v)), float(v.max()), float((v >= 60).mean())
        best = a.loc[a.groupby(["dev", "idx"]).pce.idxmax()]
        top1 = float((best.dev == best.cand).mean())
        per_dev = {t_: float((best[best.dev == t_].dev == best[best.dev == t_].cand).mean()) for t_ in order}
        nphoto = {t_: int(a[a.dev == t_].idx.nunique()) for t_ in order}
        true = a[a.dev == a.cand].pce
        false = a[a.dev != a.cand].pce
        out[arm] = dict(name=nice, desc=desc, median_pce=med.tolist(), max_pce=mx.tolist(), share_ge60=share.tolist(), top1=top1, per_device_top1=per_dev,
                        n_photos=nphoto, n_photos_total=int(len(best)), median_true=float(true.median()), median_false=float(false.median()),
                        tpr60=float((true >= 60).mean()), n_false_pairs=int(len(false)), false_ge60=int((false >= 60).sum()), max_false=float(false.max()))
    return dict(devices=order, groups=groups, arms=out, threshold=60, source=F021C + "; accuracy and PCE-60 summary cross-check: " + F021 + " ('(ii) VISION devices')",
                sentence="Each photo's sensor-noise residual is correlated with every device's reference fingerprint; the true device lights up on the diagonal (median PCE about 1,000 on native photos) while wrong devices stay near 1-3. Platform copies are weaker but the right device still wins most of the time.")


def live_model():
    p = C.MODELS / "meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    a, d = m["attribution"], m["detector"]
    return dict(fold=a["fold"], n_test=a["n_test"], heldout_bacc=a["heldout_bacc_fold"], reference_5fold=a["reference_5fold"], msp_threshold=a["msp_threshold"],
                uncertain_on_heldout=a["uncertain_on_heldout"], detector_threshold_prob=d["threshold_prob"], detector_in_source=d["in_source_heldout"],
                source="poc/fingerprint-demo/models/meta.json (train_models.py)")


def main():
    out = dict(scorecard=scorecard(), laundering=laundering(), layers=layers(), detector_embeddings=detector_emb(), learning_curve=learning_curve(),
               unseen_fpr=unseen_fpr(), open_set=open_set(), near_twins=near_twins(), prnu=prnu(), live_model=live_model())
    (HERE / "findings.json").write_text(json.dumps(out, indent=1))
    n = {a: round(v["top1"], 3) for a, v in out["prnu"]["arms"].items()}
    print("wrote findings.json; PRNU top-1 recomputed from csv:", n)


if __name__ == "__main__":
    main()
