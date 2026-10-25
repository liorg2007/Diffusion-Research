# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 27.4% [26.7%, 28.3%] | 19.9% [19.4%, 20.7%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 36.5% [35.9%, 37.7%] | 27.3% [26.6%, 28.6%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 47.8% [46.9%, 49.6%] | 39.5% [38.5%, 41.4%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 74.4% [74.1%, 75.0%] | 68.3% [68.0%, 68.9%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 15.5% [15.1%, 16.3%] | 8.1% [7.8%, 8.8%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 26.5% [25.8%, 27.6%] | 19.3% [18.8%, 20.3%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 38.4% [37.3%, 40.3%] | 29.4% [28.2%, 31.5%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 69.1% [68.3%, 69.9%] | 62.6% [61.7%, 63.6%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 42.7% [42.7%, 42.7%] | 35.7% [35.7%, 35.7%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.84% | 2.23% | 1.68% | 1.40% | 0.56% |
| pooled@0.01 | 0.28% | 2.51% | 1.12% | 1.40% | 1.12% |
| wg@0.01 | 0.28% | 1.96% | 0.56% | 0.84% | 0.28% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 15.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 26.5% | 0.10% [0.00%, 0.16%] | 0.12% [-0.08%, 0.30%] | 0.00 |
| 2% | 38.4% | 0.36% [0.00%, 1.00%] | 0.18% [0.00%, 0.42%] | 0.00 |
| 5% | 69.1% | -0.30% [-0.81%, 0.33%] | -0.01% [-0.18%, 0.16%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 23.0% [22.6%, 23.6%] | 13.0% [12.4%, 13.9%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 29.0% [27.7%, 31.7%] | 20.7% [19.3%, 23.3%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 50.5% [49.4%, 52.2%] | 41.6% [40.5%, 43.4%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 70.3% [70.0%, 70.8%] | 62.3% [61.9%, 62.9%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 16.1% [15.9%, 16.3%] | 6.8% [6.7%, 7.0%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 24.3% [23.5%, 25.8%] | 14.1% [13.2%, 15.8%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 35.1% [33.7%, 37.7%] | 28.1% [26.8%, 30.3%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 67.4% [67.1%, 67.9%] | 59.2% [59.0%, 59.8%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 56.2% [56.2%, 56.2%] | 46.0% [46.0%, 46.0%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.40% | 2.79% | 2.23% | 3.07% | 1.96% |
| pooled@0.01 | 0.56% | 2.51% | 1.68% | 1.68% | 1.40% |
| wg@0.01 | 0.56% | 1.96% | 1.40% | 1.40% | 0.56% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 16.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 24.3% | 0.28% [0.00%, 0.42%] | 0.17% [-0.17%, 0.43%] | 0.00 |
| 2% | 35.1% | 0.00% [-0.17%, 0.00%] | -0.06% [-0.19%, 0.08%] | 0.00 |
| 5% | 67.4% | 0.11% [-0.82%, 0.33%] | -0.24% [-0.51%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 60.1% [59.8%, 60.5%] | 11.3% [11.1%, 11.5%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 66.7% [66.4%, 67.1%] | 15.1% [15.0%, 15.3%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 77.3% [77.0%, 77.6%] | 23.3% [22.8%, 23.8%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 86.6% [86.4%, 86.8%] | 40.3% [40.1%, 40.5%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 32.1% [31.1%, 34.2%] | 4.0% [3.8%, 4.4%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 59.6% [59.4%, 59.9%] | 11.0% [10.9%, 11.2%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 64.6% [63.8%, 65.6%] | 15.2% [15.0%, 15.5%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 85.8% [85.7%, 86.0%] | 38.5% [38.2%, 38.9%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 85.2% [85.2%, 85.2%] | 35.1% [35.1%, 35.1%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 3.35% | 4.47% | 2.79% | 4.19% | 4.19% |
| pooled@0.01 | 0.00% | 3.07% | 0.28% | 0.84% | 0.56% |
| wg@0.01 | 0.00% | 2.79% | 0.00% | 0.28% | 0.00% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 32.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 59.6% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 64.6% | 0.23% [0.00%, 0.76%] | 0.04% [-0.03%, 0.15%] | 0.00 |
| 5% | 85.8% | -0.84% [-1.96%, 0.01%] | -0.22% [-0.50%, -0.00%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 12.3% [12.1%, 12.6%] | 13.8% [13.6%, 14.1%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 18.6% [18.5%, 18.8%] | 20.3% [20.2%, 20.4%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 25.2% [24.8%, 25.7%] | 26.4% [26.0%, 26.8%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 44.2% [43.9%, 44.4%] | 46.1% [45.8%, 46.4%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 6.3% [6.3%, 6.4%] | 6.5% [6.5%, 6.6%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 12.3% [12.0%, 12.8%] | 13.6% [13.4%, 14.1%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 19.2% [18.9%, 19.6%] | 20.6% [20.3%, 20.9%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 34.5% [34.1%, 35.1%] | 36.4% [36.0%, 37.1%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 31.2% [31.2%, 31.2%] | 32.3% [32.3%, 32.3%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 4.19% | 1.40% | 1.96% | 38.27% | 9.78% |
| pooled@0.01 | 2.51% | 0.28% | 0.28% | 27.37% | 6.70% |
| wg@0.01 | 2.51% | 0.00% | 0.00% | 12.57% | 3.07% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 6.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 12.3% | -2.89% [-4.93%, -1.13%] | -0.51% [-0.92%, -0.19%] | 0.00 |
| 2% | 19.2% | -2.17% [-3.34%, -1.05%] | -0.38% [-0.69%, -0.06%] | 0.00 |
| 5% | 34.5% | -1.95% [-3.38%, -0.72%] | -0.37% [-0.77%, 0.01%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 76.6% [75.9%, 77.8%] | 19.2% [18.5%, 20.4%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 87.9% [87.6%, 88.3%] | 29.8% [29.3%, 30.6%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 93.5% [93.4%, 93.7%] | 39.7% [39.2%, 40.2%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 97.9% [97.8%, 98.0%] | 54.9% [54.6%, 55.2%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 68.3% [67.9%, 69.1%] | 12.9% [12.6%, 13.5%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 74.0% [73.3%, 75.5%] | 17.6% [16.8%, 19.1%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 85.7% [85.1%, 86.6%] | 27.8% [26.9%, 29.0%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 96.4% [96.3%, 96.5%] | 47.8% [47.3%, 48.4%] |
| M1 | 2.29% [0.84%, 4.28%] | 3.91% [1.69%, 7.02%] | 92.0% [92.0%, 92.0%] | 37.1% [37.1%, 37.1%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 2.23% | 1.12% | 1.12% | 3.07% | 3.91% |
| pooled@0.01 | 2.23% | 0.56% | 0.28% | 1.40% | 2.51% |
| wg@0.01 | 1.68% | 0.28% | 0.28% | 0.84% | 1.40% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 68.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 74.0% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 85.7% | 2.32% [0.65%, 3.82%] | 0.52% [0.11%, 0.95%] | 0.00 |
| 5% | 96.4% | -1.40% [-2.54%, 0.17%] | -0.36% [-0.70%, -0.08%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 51.7% [51.5%, 52.0%] | 26.6% [26.2%, 27.3%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 55.2% [54.9%, 55.6%] | 34.3% [33.7%, 35.0%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 63.5% [63.4%, 63.6%] | 49.8% [49.6%, 50.0%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 68.9% [68.8%, 69.0%] | 62.9% [62.8%, 63.1%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 46.7% [46.5%, 47.1%] | 16.5% [16.1%, 17.0%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 52.5% [52.3%, 52.7%] | 27.1% [26.6%, 27.7%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 56.1% [55.8%, 56.7%] | 35.8% [35.2%, 36.9%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 65.5% [65.3%, 65.6%] | 54.6% [54.4%, 54.8%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 54.3% [54.3%, 54.3%] | 32.6% [32.6%, 32.6%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.96% | 0.00% | 0.56% | 0.00% | 0.00% |
| pooled@0.01 | 2.51% | 0.00% | 0.56% | 0.00% | 0.28% |
| wg@0.01 | 2.23% | 0.00% | 0.28% | 0.00% | 0.00% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 46.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 52.5% | 0.15% [0.00%, 0.46%] | 0.00% [-0.07%, 0.08%] | 0.00 |
| 2% | 56.1% | 1.97% [0.65%, 3.64%] | 0.32% [0.08%, 0.66%] | 0.00 |
| 5% | 65.5% | 0.38% [-0.28%, 1.28%] | 0.01% [-0.19%, 0.21%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
