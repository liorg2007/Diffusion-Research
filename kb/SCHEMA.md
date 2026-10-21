# KB schema (all entries in plain language; every number must be traceable)

## Grounding rules
- Every number comes from the paper/repo, with the table/figure it came from, e.g. "AUROC 0.93 (Table 3, Synthbuster)".
- If you could not verify it from the primary source, append **[unverified]**. Never invent citations, numbers, URLs, or licenses.
- Prefer peer-reviewed versions; give arXiv ID AND venue if both exist.

## kb/papers/<id>.md   (id = lowercase short name + venue/year, e.g. `npr-cvpr2024`, `raid-arxiv2025`)
```
# <Full paper title>
- **id:** <id>
- **Authors / venue / year:** ...
- **Links:** arXiv <id> · paper URL · code URL
- **Family:** low-level | foundation-features | reconstruction | data-bias-benchmarks | robustness-attacks
- **Status:** read-full | read-abstract-only | partially-verified

## Summary (plain language)
2–4 sentences a newcomer can follow.

## How it works
Inputs → features → classifier; what is trained, what is frozen; key hyper-parameters.

## Training data
Which generators / real sources / sizes / preprocessing the released model was trained on.

## Reported results
Exact numbers with dataset + table reference. Separate: in-distribution vs unseen generators vs perturbed (JPEG/resize/social).
Note whether FPR at a fixed threshold / calibration on fresh real photos is reported (usually it is not).

## Known weaknesses / failure modes
From the paper itself AND from later papers that evaluated it (cite them).

## Code + weights
URL, license (exact SPDX or text, note NON-COMMERCIAL), weight file names, whether checkpoints are publicly downloadable.

## Verified?
yes/no + how (e.g. "numbers copied from Table 2 of arXiv v2 PDF"; "license read from repo LICENSE file").

## Relevance to this project
1–3 bullets: what it tells us about generalization to unseen generators / in-the-wild images.
```

## kb/datasets/<name>.md
```
# <Dataset name>
- Source/URL · paper · License (note non-commercial) · Size (images, GB)
- Real sources · Generators (list) · Formats (PNG/JPEG, resolutions) — format bias risk!
- Download procedure · Known issues (leakage, JPEG/PNG bias, size bias) · Used in our experiments? (yes/no)
```

## kb/ideas/<id>.md
```
# <Idea title>
- **Hypothesis:** one falsifiable sentence
- **Rationale:**
- **Closest prior work:** (kb ids) and how this differs
- **Evidence for / against:**
- **Test plan (Mac-feasible):** data, model, metric, pass/fail threshold
- **Status:** proposed | testing | supported | refuted | abandoned
```

## kb/index.md row format
`| id | title | family | 1-line takeaway | status |`
