# H2-v2 tables (scores_long)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

PILOT data (VISION + 6 Synthbuster): not a Phase 4 verdict.

Brand derivable from src_path: **True**. Groupings: platform_brand, platform.

## A0 / grouping platform_brand  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH/all (n=358, 25 dev); WA/all (n=358, 25 dev); native/all (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.07% [0.01%, 3.04%] | 1.30% [0.05%, 3.75%] | 46.9% [46.1%, 48.5%] | 42.2% [41.4%, 43.8%] |
| pooled@0.01 | 1.71% [0.01%, 4.87%] | 2.00% [0.05%, 5.73%] | 52.8% [51.7%, 54.9%] | 47.6% [46.4%, 50.1%] |
| pooled@0.02 | 2.77% [0.23%, 7.23%] | 2.93% [0.23%, 7.66%] | 71.6% [70.7%, 73.0%] | 69.1% [68.1%, 70.8%] |
| pooled@0.05 | 5.07% [1.30%, 10.40%] | 5.59% [1.60%, 11.14%] | 87.6% [87.1%, 88.1%] | 87.4% [87.0%, 87.8%] |
| wg@0.005 | 0.40% [0.00%, 1.13%] | 0.65% [0.00%, 1.85%] | 42.2% [42.0%, 42.7%] | 35.3% [35.0%, 36.0%] |
| wg@0.01 | 1.55% [0.01%, 4.41%] | 1.63% [0.05%, 4.71%] | 49.3% [48.1%, 51.6%] | 46.0% [44.8%, 48.3%] |
| wg@0.02 | 2.50% [0.21%, 6.64%] | 2.79% [0.23%, 7.25%] | 64.8% [63.4%, 66.8%] | 60.5% [58.8%, 62.8%] |
| wg@0.05 | 4.89% [1.21%, 10.17%] | 5.35% [1.51%, 10.80%] | 86.8% [86.4%, 87.4%] | 86.3% [85.9%, 86.8%] |
| M1 | 1.45% [0.20%, 3.59%] | 1.49% [0.23%, 3.92%] | 67.4% [67.4%, 67.4%] | 63.5% [63.5%, 63.5%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.49% | 1.44% | 1.44% | 1.49% | 1.40% |
| pooled@0.01 | 1.63% | 1.58% | 1.49% | 2.00% | 1.86% |
| wg@0.01 | 1.58% | 1.58% | 1.44% | 1.63% | 1.49% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 42.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 49.3% | 0.11% [-0.05%, 0.31%] | 0.23% [-0.01%, 0.50%] | 0.00 |
| 2% | 64.8% | 0.02% [0.00%, 0.03%] | 0.02% [0.00%, 0.03%] | 0.00 |
| 5% | 86.8% | 0.21% [-0.01%, 0.35%] | 0.15% [0.08%, 0.21%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/group/jpeg75_resize05/all)

## A0 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.07% [0.01%, 3.04%] | 1.30% [0.05%, 3.75%] | 46.9% [46.1%, 48.5%] | 42.2% [41.4%, 43.8%] |
| pooled@0.01 | 1.71% [0.01%, 4.87%] | 2.00% [0.05%, 5.73%] | 52.8% [51.7%, 54.9%] | 47.6% [46.4%, 50.1%] |
| pooled@0.02 | 2.77% [0.23%, 7.23%] | 2.93% [0.23%, 7.66%] | 71.6% [70.7%, 73.0%] | 69.1% [68.1%, 70.8%] |
| pooled@0.05 | 5.07% [1.30%, 10.40%] | 5.59% [1.60%, 11.14%] | 87.6% [87.1%, 88.1%] | 87.4% [87.0%, 87.8%] |
| wg@0.005 | 0.40% [0.00%, 1.13%] | 0.65% [0.00%, 1.85%] | 42.2% [42.0%, 42.7%] | 35.3% [35.0%, 36.0%] |
| wg@0.01 | 1.55% [0.01%, 4.41%] | 1.63% [0.05%, 4.71%] | 49.3% [48.1%, 51.6%] | 46.0% [44.8%, 48.3%] |
| wg@0.02 | 2.50% [0.21%, 6.64%] | 2.79% [0.23%, 7.25%] | 64.8% [63.4%, 66.8%] | 60.5% [58.8%, 62.8%] |
| wg@0.05 | 4.89% [1.21%, 10.17%] | 5.35% [1.51%, 10.80%] | 86.8% [86.4%, 87.4%] | 86.3% [85.9%, 86.8%] |
| M1 | 1.45% [0.20%, 3.59%] | 1.49% [0.23%, 3.92%] | 67.4% [67.4%, 67.4%] | 63.5% [63.5%, 63.5%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.49% | 1.44% | 1.44% | 1.49% | 1.40% |
| pooled@0.01 | 1.63% | 1.58% | 1.49% | 2.00% | 1.86% |
| wg@0.01 | 1.58% | 1.58% | 1.44% | 1.63% | 1.49% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 42.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 49.3% | 0.11% [-0.05%, 0.31%] | 0.23% [-0.01%, 0.50%] | 0.00 |
| 2% | 64.8% | 0.02% [0.00%, 0.03%] | 0.02% [0.00%, 0.03%] | 0.00 |
| 5% | 86.8% | 0.21% [-0.01%, 0.35%] | 0.15% [0.08%, 0.21%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/group/jpeg75_resize05)

## A1 / grouping platform_brand  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH/all (n=358, 25 dev); WA/all (n=358, 25 dev); native/all (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.98% [0.00%, 2.68%] | 1.26% [0.00%, 3.53%] | 64.7% [64.0%, 65.8%] | 62.9% [62.1%, 64.2%] |
| pooled@0.01 | 1.28% [0.03%, 3.46%] | 1.58% [0.09%, 4.33%] | 72.0% [71.7%, 72.7%] | 71.5% [71.1%, 72.2%] |
| pooled@0.02 | 2.11% [0.35%, 5.08%] | 2.23% [0.51%, 5.51%] | 78.2% [77.9%, 78.9%] | 78.1% [77.7%, 78.7%] |
| pooled@0.05 | 5.17% [1.62%, 9.76%] | 5.63% [1.96%, 10.66%] | 88.9% [88.6%, 89.3%] | 89.1% [88.8%, 89.4%] |
| wg@0.005 | 0.26% [0.00%, 0.61%] | 0.33% [0.00%, 0.84%] | 55.3% [54.9%, 55.9%] | 53.4% [53.0%, 53.9%] |
| wg@0.01 | 1.15% [0.00%, 3.16%] | 1.54% [0.00%, 4.32%] | 70.2% [69.8%, 70.9%] | 68.3% [67.7%, 69.2%] |
| wg@0.02 | 1.87% [0.25%, 4.66%] | 2.05% [0.42%, 5.17%] | 76.2% [75.8%, 76.9%] | 75.8% [75.4%, 76.6%] |
| wg@0.05 | 4.93% [1.53%, 9.41%] | 5.49% [1.85%, 10.45%] | 88.2% [87.9%, 88.7%] | 88.4% [88.2%, 88.8%] |
| M1 | 0.94% [0.08%, 2.33%] | 1.21% [0.14%, 3.05%] | 71.4% [71.4%, 71.4%] | 71.1% [71.1%, 71.1%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.21% | 1.16% | 0.88% | 0.65% | 0.79% |
| pooled@0.01 | 1.58% | 1.35% | 1.21% | 1.07% | 1.21% |
| wg@0.01 | 1.54% | 1.30% | 1.16% | 0.84% | 0.93% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 55.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 70.2% | 0.03% [0.00%, 0.08%] | -0.02% [-0.04%, 0.00%] | 0.00 |
| 2% | 76.2% | 0.05% [-0.05%, 0.11%] | 0.02% [-0.01%, 0.06%] | 0.00 |
| 5% | 88.2% | 0.14% [-0.08%, 0.41%] | 0.05% [-0.03%, 0.15%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): False (none)

## A1 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.98% [0.00%, 2.68%] | 1.26% [0.00%, 3.53%] | 64.7% [64.0%, 65.8%] | 62.9% [62.1%, 64.2%] |
| pooled@0.01 | 1.28% [0.03%, 3.46%] | 1.58% [0.09%, 4.33%] | 72.0% [71.7%, 72.7%] | 71.5% [71.1%, 72.2%] |
| pooled@0.02 | 2.11% [0.35%, 5.08%] | 2.23% [0.51%, 5.51%] | 78.2% [77.9%, 78.9%] | 78.1% [77.7%, 78.7%] |
| pooled@0.05 | 5.17% [1.62%, 9.76%] | 5.63% [1.96%, 10.66%] | 88.9% [88.6%, 89.3%] | 89.1% [88.8%, 89.4%] |
| wg@0.005 | 0.26% [0.00%, 0.61%] | 0.33% [0.00%, 0.84%] | 55.3% [54.9%, 55.9%] | 53.4% [53.0%, 53.9%] |
| wg@0.01 | 1.15% [0.00%, 3.16%] | 1.54% [0.00%, 4.32%] | 70.2% [69.8%, 70.9%] | 68.3% [67.7%, 69.2%] |
| wg@0.02 | 1.87% [0.25%, 4.66%] | 2.05% [0.42%, 5.17%] | 76.2% [75.8%, 76.9%] | 75.8% [75.4%, 76.6%] |
| wg@0.05 | 4.93% [1.53%, 9.41%] | 5.49% [1.85%, 10.45%] | 88.2% [87.9%, 88.7%] | 88.4% [88.2%, 88.8%] |
| M1 | 0.94% [0.08%, 2.33%] | 1.21% [0.14%, 3.05%] | 71.4% [71.4%, 71.4%] | 71.1% [71.1%, 71.1%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.21% | 1.16% | 0.88% | 0.65% | 0.79% |
| pooled@0.01 | 1.58% | 1.35% | 1.21% | 1.07% | 1.21% |
| wg@0.01 | 1.54% | 1.30% | 1.16% | 0.84% | 0.93% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 55.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 70.2% | 0.03% [0.00%, 0.08%] | -0.02% [-0.04%, 0.00%] | 0.00 |
| 2% | 76.2% | 0.05% [-0.05%, 0.11%] | 0.02% [-0.01%, 0.06%] | 0.00 |
| 5% | 88.2% | 0.14% [-0.08%, 0.41%] | 0.05% [-0.03%, 0.15%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): False (none)
