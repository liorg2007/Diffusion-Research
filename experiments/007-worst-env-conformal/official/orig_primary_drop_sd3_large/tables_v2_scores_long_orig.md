# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 29.3% [28.6%, 30.3%] | 22.8% [22.3%, 23.7%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 38.9% [38.2%, 40.1%] | 30.7% [30.0%, 32.0%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 50.3% [49.4%, 52.0%] | 42.6% [41.6%, 44.5%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 76.2% [75.9%, 76.7%] | 70.9% [70.6%, 71.4%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 17.6% [17.2%, 18.4%] | 9.8% [9.4%, 10.5%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 28.3% [27.6%, 29.4%] | 22.3% [21.7%, 23.3%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 40.7% [39.6%, 42.6%] | 32.8% [31.6%, 34.8%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 71.0% [70.3%, 71.8%] | 65.4% [64.5%, 66.4%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 45.3% [45.3%, 45.3%] | 38.9% [38.9%, 38.9%] |

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
| 0.5% | 17.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 28.3% | 0.11% [0.00%, 0.18%] | 0.14% [-0.07%, 0.32%] | 0.00 |
| 2% | 40.7% | 0.36% [0.00%, 1.01%] | 0.19% [0.00%, 0.45%] | 0.00 |
| 5% | 71.0% | -0.30% [-0.82%, 0.33%] | -0.01% [-0.19%, 0.16%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 25.2% [24.8%, 25.8%] | 16.2% [15.6%, 17.1%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 31.8% [30.5%, 34.3%] | 24.1% [22.6%, 26.8%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 52.0% [50.9%, 53.7%] | 45.7% [44.5%, 47.4%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 71.9% [71.6%, 72.3%] | 65.8% [65.5%, 66.4%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 17.9% [17.8%, 18.1%] | 8.5% [8.3%, 8.7%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 26.5% [25.7%, 28.0%] | 17.6% [16.7%, 19.3%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 37.3% [35.9%, 39.8%] | 31.3% [30.0%, 33.7%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 68.9% [68.6%, 69.3%] | 63.0% [62.7%, 63.5%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 57.6% [57.6%, 57.6%] | 50.6% [50.6%, 50.6%] |

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
| 0.5% | 17.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 26.5% | 0.29% [0.00%, 0.48%] | 0.17% [-0.17%, 0.47%] | 0.00 |
| 2% | 37.3% | 0.00% [-0.17%, 0.00%] | -0.06% [-0.19%, 0.08%] | 0.00 |
| 5% | 68.9% | 0.11% [-0.80%, 0.34%] | -0.23% [-0.49%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 61.0% [60.7%, 61.3%] | 14.2% [14.0%, 14.4%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 67.3% [67.0%, 67.7%] | 18.8% [18.6%, 19.0%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 77.8% [77.5%, 78.0%] | 29.0% [28.5%, 29.7%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 86.8% [86.6%, 87.0%] | 47.8% [47.5%, 48.0%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 34.0% [33.0%, 36.0%] | 5.7% [5.4%, 6.2%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 60.5% [60.3%, 60.8%] | 14.0% [13.9%, 14.2%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 65.5% [64.7%, 66.3%] | 18.9% [18.5%, 19.3%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 86.0% [85.8%, 86.2%] | 46.0% [45.6%, 46.4%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 85.4% [85.4%, 85.4%] | 42.4% [42.4%, 42.4%] |

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
| 0.5% | 34.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 60.5% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 65.5% | 0.22% [0.00%, 0.71%] | 0.03% [-0.04%, 0.14%] | 0.00 |
| 5% | 86.0% | -0.84% [-1.96%, 0.01%] | -0.23% [-0.50%, -0.00%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 9.4% [9.3%, 9.6%] | 15.6% [15.4%, 15.9%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 14.2% [14.1%, 14.4%] | 23.2% [23.0%, 23.4%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 19.9% [19.6%, 20.3%] | 30.1% [29.6%, 30.7%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 37.3% [37.1%, 37.5%] | 49.8% [49.6%, 50.0%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 5.2% [5.1%, 5.2%] | 7.4% [7.3%, 7.4%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 9.5% [9.4%, 9.9%] | 15.3% [15.0%, 15.9%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 14.7% [14.4%, 15.0%] | 23.7% [23.4%, 24.1%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 28.0% [27.6%, 28.6%] | 41.0% [40.6%, 41.6%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 24.8% [24.8%, 24.8%] | 37.4% [37.4%, 37.4%] |

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
| 0.5% | 5.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 9.5% | -3.57% [-5.69%, -1.75%] | -0.69% [-1.11%, -0.33%] | 0.00 |
| 2% | 14.7% | -2.11% [-3.28%, -0.98%] | -0.36% [-0.67%, -0.04%] | 0.00 |
| 5% | 28.0% | -1.89% [-3.27%, -0.68%] | -0.34% [-0.73%, 0.04%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 81.2% [80.6%, 82.1%] | 22.5% [21.6%, 24.1%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 89.8% [89.6%, 90.2%] | 36.2% [35.7%, 37.1%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 94.3% [94.2%, 94.5%] | 47.5% [47.0%, 48.1%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 98.4% [98.3%, 98.5%] | 63.2% [63.0%, 63.6%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 74.2% [73.8%, 74.9%] | 14.9% [14.5%, 15.7%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 79.1% [78.6%, 80.2%] | 20.2% [19.2%, 22.1%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 88.3% [87.8%, 89.0%] | 33.9% [32.9%, 35.3%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 96.9% [96.8%, 97.1%] | 56.0% [55.5%, 56.6%] |
| M1 | 2.29% [0.84%, 4.28%] | 3.91% [1.69%, 7.02%] | 93.0% [93.0%, 93.0%] | 44.6% [44.6%, 44.6%] |

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
| 0.5% | 74.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 79.1% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 88.3% | 2.31% [0.63%, 3.81%] | 0.51% [0.11%, 0.93%] | 0.00 |
| 5% | 96.9% | -1.31% [-2.38%, 0.19%] | -0.31% [-0.63%, -0.04%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 55.6% [55.4%, 55.9%] | 31.4% [31.0%, 32.0%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 58.9% [58.6%, 59.2%] | 39.2% [38.6%, 40.0%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 66.0% [65.9%, 66.1%] | 54.6% [54.4%, 54.8%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 70.7% [70.6%, 70.7%] | 67.0% [66.8%, 67.2%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 51.6% [51.4%, 51.8%] | 20.6% [20.2%, 21.2%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 56.3% [56.1%, 56.5%] | 31.9% [31.4%, 32.5%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 59.7% [59.4%, 60.2%] | 40.7% [40.0%, 41.8%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 67.7% [67.5%, 67.8%] | 59.1% [58.9%, 59.3%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 58.0% [58.0%, 58.0%] | 37.6% [37.6%, 37.6%] |

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
| 0.5% | 51.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 56.3% | 0.16% [0.00%, 0.50%] | 0.01% [-0.06%, 0.09%] | 0.00 |
| 2% | 59.7% | 1.93% [0.64%, 3.58%] | 0.31% [0.07%, 0.64%] | 0.00 |
| 5% | 67.7% | 0.46% [-0.23%, 1.40%] | 0.05% [-0.14%, 0.26%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
