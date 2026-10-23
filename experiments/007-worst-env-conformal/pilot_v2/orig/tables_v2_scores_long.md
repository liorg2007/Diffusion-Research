# H2-v2 tables (scores_long)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

PILOT data (VISION + 6 Synthbuster): not a Phase 4 verdict.

Brand derivable from src_path: **True**. Groupings: platform_brand, platform.

## A0 / grouping platform_brand  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH/all (n=358, 25 dev); WA/all (n=358, 25 dev); native/all (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.74% [0.35%, 1.21%] | 2.51% [1.27%, 3.98%] | 69.0% [68.5%, 69.5%] | 74.3% [73.8%, 74.8%] |
| pooled@0.01 | 1.06% [0.52%, 1.71%] | 3.31% [1.78%, 5.32%] | 77.4% [77.2%, 77.6%] | 81.8% [81.6%, 82.0%] |
| pooled@0.02 | 2.27% [1.27%, 3.55%] | 5.82% [3.42%, 9.14%] | 84.4% [84.2%, 84.6%] | 89.0% [88.8%, 89.3%] |
| pooled@0.05 | 6.12% [3.93%, 8.61%] | 15.18% [11.02%, 20.13%] | 91.2% [91.1%, 91.3%] | 95.3% [95.2%, 95.4%] |
| wg@0.005 | 0.41% [0.17%, 0.70%] | 1.77% [0.79%, 2.79%] | 51.6% [51.3%, 52.2%] | 54.1% [53.8%, 54.8%] |
| wg@0.01 | 0.74% [0.36%, 1.18%] | 2.84% [1.49%, 4.47%] | 66.1% [65.5%, 66.9%] | 71.8% [71.2%, 72.5%] |
| wg@0.02 | 1.23% [0.64%, 1.94%] | 4.33% [2.28%, 7.03%] | 76.7% [76.4%, 77.0%] | 81.0% [80.8%, 81.2%] |
| wg@0.05 | 3.99% [2.63%, 5.59%] | 11.69% [8.33%, 15.73%] | 88.0% [87.9%, 88.2%] | 92.8% [92.7%, 93.0%] |
| M1 | 0.76% [0.29%, 1.35%] | 1.68% [0.70%, 2.83%] | 74.3% [74.3%, 74.3%] | 79.6% [79.6%, 79.6%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.33% | 1.68% | 0.37% | 1.02% | 0.42% |
| pooled@0.01 | 0.05% | 3.31% | 0.23% | 1.30% | 0.42% |
| wg@0.01 | 0.00% | 2.84% | 0.05% | 0.56% | 0.23% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 51.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 66.1% | 0.41% [0.08%, 0.80%] | 0.05% [-0.02%, 0.14%] | 0.00 |
| 2% | 76.7% | 1.12% [0.48%, 2.01%] | 0.21% [0.09%, 0.38%] | 0.00 |
| 5% | 88.0% | 2.53% [1.68%, 3.50%] | 0.38% [0.23%, 0.54%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/WA, pooled@0.01/group/WA/all)

## A0 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.74% [0.35%, 1.21%] | 2.51% [1.27%, 3.98%] | 69.0% [68.5%, 69.5%] | 74.3% [73.8%, 74.8%] |
| pooled@0.01 | 1.06% [0.52%, 1.71%] | 3.31% [1.78%, 5.32%] | 77.4% [77.2%, 77.6%] | 81.8% [81.6%, 82.0%] |
| pooled@0.02 | 2.27% [1.27%, 3.55%] | 5.82% [3.42%, 9.14%] | 84.4% [84.2%, 84.6%] | 89.0% [88.8%, 89.3%] |
| pooled@0.05 | 6.12% [3.93%, 8.61%] | 15.18% [11.02%, 20.13%] | 91.2% [91.1%, 91.3%] | 95.3% [95.2%, 95.4%] |
| wg@0.005 | 0.41% [0.17%, 0.70%] | 1.77% [0.79%, 2.79%] | 51.6% [51.3%, 52.2%] | 54.1% [53.8%, 54.8%] |
| wg@0.01 | 0.74% [0.36%, 1.18%] | 2.84% [1.49%, 4.47%] | 66.1% [65.5%, 66.9%] | 71.8% [71.2%, 72.5%] |
| wg@0.02 | 1.23% [0.64%, 1.94%] | 4.33% [2.28%, 7.03%] | 76.7% [76.4%, 77.0%] | 81.0% [80.8%, 81.2%] |
| wg@0.05 | 3.99% [2.63%, 5.59%] | 11.69% [8.33%, 15.73%] | 88.0% [87.9%, 88.2%] | 92.8% [92.7%, 93.0%] |
| M1 | 0.76% [0.29%, 1.35%] | 1.68% [0.70%, 2.83%] | 74.3% [74.3%, 74.3%] | 79.6% [79.6%, 79.6%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.33% | 1.68% | 0.37% | 1.02% | 0.42% |
| pooled@0.01 | 0.05% | 3.31% | 0.23% | 1.30% | 0.42% |
| wg@0.01 | 0.00% | 2.84% | 0.05% | 0.56% | 0.23% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 51.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 66.1% | 0.41% [0.08%, 0.80%] | 0.05% [-0.02%, 0.14%] | 0.00 |
| 2% | 76.7% | 1.12% [0.48%, 2.01%] | 0.21% [0.09%, 0.38%] | 0.00 |
| 5% | 88.0% | 2.53% [1.68%, 3.50%] | 0.38% [0.23%, 0.54%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/WA, pooled@0.01/group/WA)

## A1 / grouping platform_brand  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH/all (n=358, 25 dev); WA/all (n=358, 25 dev); native/all (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.64% [0.22%, 1.13%] | 1.96% [0.56%, 3.76%] | 69.6% [69.1%, 70.2%] | 74.7% [74.2%, 75.3%] |
| pooled@0.01 | 1.09% [0.44%, 1.91%] | 2.84% [1.03%, 5.40%] | 77.4% [77.1%, 77.8%] | 82.0% [81.7%, 82.3%] |
| pooled@0.02 | 2.31% [1.06%, 3.94%] | 4.14% [2.09%, 7.47%] | 83.6% [83.4%, 83.9%] | 88.0% [87.8%, 88.2%] |
| pooled@0.05 | 5.75% [3.11%, 9.07%] | 10.43% [6.45%, 15.20%] | 90.2% [90.1%, 90.4%] | 94.1% [94.0%, 94.3%] |
| wg@0.005 | 0.18% [0.00%, 0.45%] | 0.84% [0.00%, 2.07%] | 50.1% [49.6%, 50.9%] | 53.8% [53.3%, 54.7%] |
| wg@0.01 | 0.56% [0.18%, 1.04%] | 2.05% [0.61%, 3.85%] | 68.2% [67.7%, 68.8%] | 73.2% [72.6%, 73.8%] |
| wg@0.02 | 1.23% [0.49%, 2.12%] | 3.35% [1.26%, 6.08%] | 76.7% [76.3%, 77.2%] | 81.9% [81.6%, 82.3%] |
| wg@0.05 | 4.13% [2.24%, 6.50%] | 8.24% [4.78%, 12.36%] | 87.8% [87.7%, 88.0%] | 91.6% [91.5%, 91.8%] |
| M1 | 1.11% [0.51%, 1.80%] | 1.96% [1.04%, 3.59%] | 77.8% [77.8%, 77.8%] | 82.8% [82.8%, 82.8%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.51% | 1.96% | 0.74% | 1.77% | 0.56% |
| pooled@0.01 | 0.37% | 2.84% | 0.19% | 1.44% | 0.61% |
| wg@0.01 | 0.28% | 2.05% | 0.00% | 0.28% | 0.19% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 50.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 68.2% | 0.13% [0.02%, 0.29%] | -0.00% [-0.04%, 0.03%] | 0.00 |
| 2% | 76.7% | 0.64% [0.17%, 1.13%] | 0.19% [0.05%, 0.35%] | 0.00 |
| 5% | 87.8% | 1.12% [-0.60%, 1.65%] | 0.08% [-0.07%, 0.22%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/WA, pooled@0.01/group/WA/all)

## A1 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.64% [0.22%, 1.13%] | 1.96% [0.56%, 3.76%] | 69.6% [69.1%, 70.2%] | 74.7% [74.2%, 75.3%] |
| pooled@0.01 | 1.09% [0.44%, 1.91%] | 2.84% [1.03%, 5.40%] | 77.4% [77.1%, 77.8%] | 82.0% [81.7%, 82.3%] |
| pooled@0.02 | 2.31% [1.06%, 3.94%] | 4.14% [2.09%, 7.47%] | 83.6% [83.4%, 83.9%] | 88.0% [87.8%, 88.2%] |
| pooled@0.05 | 5.75% [3.11%, 9.07%] | 10.43% [6.45%, 15.20%] | 90.2% [90.1%, 90.4%] | 94.1% [94.0%, 94.3%] |
| wg@0.005 | 0.18% [0.00%, 0.45%] | 0.84% [0.00%, 2.07%] | 50.1% [49.6%, 50.9%] | 53.8% [53.3%, 54.7%] |
| wg@0.01 | 0.56% [0.18%, 1.04%] | 2.05% [0.61%, 3.85%] | 68.2% [67.7%, 68.8%] | 73.2% [72.6%, 73.8%] |
| wg@0.02 | 1.23% [0.49%, 2.12%] | 3.35% [1.26%, 6.08%] | 76.7% [76.3%, 77.2%] | 81.9% [81.6%, 82.3%] |
| wg@0.05 | 4.13% [2.24%, 6.50%] | 8.24% [4.78%, 12.36%] | 87.8% [87.7%, 88.0%] | 91.6% [91.5%, 91.8%] |
| M1 | 1.11% [0.51%, 1.80%] | 1.96% [1.04%, 3.59%] | 77.8% [77.8%, 77.8%] | 82.8% [82.8%, 82.8%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.51% | 1.96% | 0.74% | 1.77% | 0.56% |
| pooled@0.01 | 0.37% | 2.84% | 0.19% | 1.44% | 0.61% |
| wg@0.01 | 0.28% | 2.05% | 0.00% | 0.28% | 0.19% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 50.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 68.2% | 0.13% [0.02%, 0.29%] | -0.00% [-0.04%, 0.03%] | 0.00 |
| 2% | 76.7% | 0.64% [0.17%, 1.13%] | 0.19% [0.05%, 0.35%] | 0.00 |
| 5% | 87.8% | 1.12% [-0.60%, 1.65%] | 0.08% [-0.07%, 0.22%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/WA, pooled@0.01/group/WA)
