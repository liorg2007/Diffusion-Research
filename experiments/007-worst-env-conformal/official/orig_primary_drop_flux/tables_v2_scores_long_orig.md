# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 32.3% [31.6%, 33.3%] | 23.1% [22.6%, 24.1%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 42.7% [42.0%, 43.9%] | 31.5% [30.8%, 32.8%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 54.4% [53.4%, 56.1%] | 43.7% [42.7%, 45.6%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 80.3% [80.1%, 80.8%] | 72.0% [71.7%, 72.6%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 19.1% [18.6%, 19.9%] | 10.2% [9.9%, 10.9%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 31.3% [30.6%, 32.5%] | 22.7% [22.1%, 23.7%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 44.6% [43.5%, 46.5%] | 33.8% [32.6%, 35.8%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 75.0% [74.3%, 75.9%] | 66.5% [65.7%, 67.5%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 48.8% [48.8%, 48.8%] | 39.9% [39.9%, 39.9%] |

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
| 0.5% | 19.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 31.3% | 0.10% [0.00%, 0.16%] | 0.12% [-0.08%, 0.29%] | 0.00 |
| 2% | 44.6% | 0.35% [0.00%, 0.98%] | 0.17% [0.00%, 0.39%] | 0.00 |
| 5% | 75.0% | -0.29% [-0.79%, 0.36%] | 0.01% [-0.17%, 0.18%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 27.9% [27.4%, 28.6%] | 16.6% [16.0%, 17.6%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 35.1% [33.7%, 37.8%] | 24.9% [23.5%, 27.7%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 57.4% [56.3%, 59.1%] | 47.4% [46.2%, 49.1%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 76.4% [76.1%, 76.8%] | 67.5% [67.1%, 68.0%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 19.6% [19.5%, 19.9%] | 8.9% [8.7%, 9.1%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 29.4% [28.5%, 31.1%] | 17.9% [17.0%, 19.7%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 42.0% [40.5%, 44.5%] | 32.8% [31.5%, 35.1%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 73.9% [73.6%, 74.3%] | 64.3% [64.0%, 64.9%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 63.6% [63.6%, 63.6%] | 52.3% [52.3%, 52.3%] |

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
| 0.5% | 19.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 29.4% | 0.29% [0.00%, 0.47%] | 0.17% [-0.17%, 0.47%] | 0.00 |
| 2% | 42.0% | 0.00% [-0.18%, 0.00%] | -0.06% [-0.20%, 0.08%] | 0.00 |
| 5% | 73.9% | 0.10% [-0.87%, 0.32%] | -0.25% [-0.53%, -0.03%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 66.6% [66.3%, 67.0%] | 14.5% [14.3%, 14.7%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 73.5% [73.2%, 73.9%] | 19.3% [19.1%, 19.5%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 84.0% [83.7%, 84.3%] | 29.8% [29.2%, 30.4%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 91.6% [91.5%, 91.8%] | 48.5% [48.2%, 48.7%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 36.4% [35.2%, 38.6%] | 5.7% [5.4%, 6.3%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 66.0% [65.7%, 66.3%] | 14.3% [14.2%, 14.5%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 71.7% [70.9%, 72.5%] | 19.2% [18.9%, 19.7%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 91.0% [90.9%, 91.2%] | 46.5% [46.1%, 47.0%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 90.8% [90.8%, 90.8%] | 42.9% [42.9%, 42.9%] |

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
| 0.5% | 36.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 66.0% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 71.7% | 0.21% [0.00%, 0.67%] | 0.03% [-0.04%, 0.13%] | 0.00 |
| 5% | 91.0% | -0.84% [-1.96%, 0.01%] | -0.23% [-0.50%, -0.00%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 12.7% [12.5%, 13.0%] | 16.3% [16.1%, 16.7%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 19.4% [19.2%, 19.6%] | 24.0% [23.9%, 24.2%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 26.0% [25.6%, 26.5%] | 31.4% [31.0%, 32.0%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 44.7% [44.5%, 44.9%] | 52.7% [52.4%, 53.0%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 6.4% [6.4%, 6.4%] | 7.8% [7.8%, 7.8%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 12.7% [12.5%, 13.2%] | 16.0% [15.7%, 16.6%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 20.0% [19.7%, 20.4%] | 24.5% [24.2%, 25.0%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 35.0% [34.7%, 35.6%] | 42.8% [42.4%, 43.6%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 31.9% [31.9%, 31.9%] | 38.7% [38.7%, 38.7%] |

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
| 0.5% | 6.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 12.7% | -3.10% [-5.12%, -1.22%] | -0.57% [-0.96%, -0.22%] | 0.00 |
| 2% | 20.0% | -2.18% [-3.34%, -1.05%] | -0.38% [-0.69%, -0.06%] | 0.00 |
| 5% | 35.0% | -1.86% [-3.23%, -0.66%] | -0.32% [-0.72%, 0.05%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 81.3% [80.7%, 82.4%] | 23.0% [22.1%, 24.6%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 90.7% [90.5%, 91.2%] | 36.8% [36.3%, 37.7%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 95.6% [95.4%, 95.8%] | 48.1% [47.6%, 48.8%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 98.7% [98.7%, 98.8%] | 64.4% [64.1%, 64.7%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 74.3% [74.0%, 75.0%] | 15.3% [14.9%, 16.1%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 79.3% [78.7%, 80.5%] | 20.7% [19.6%, 22.6%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 88.9% [88.4%, 89.7%] | 34.6% [33.6%, 36.0%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 97.8% [97.8%, 97.9%] | 56.9% [56.4%, 57.5%] |
| M1 | 2.29% [0.84%, 4.28%] | 3.91% [1.69%, 7.02%] | 94.3% [94.3%, 94.3%] | 45.1% [45.1%, 45.1%] |

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
| 0.5% | 74.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 79.3% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 88.9% | 2.31% [0.63%, 3.81%] | 0.51% [0.11%, 0.94%] | 0.00 |
| 5% | 97.8% | -1.76% [-3.23%, 0.08%] | -0.56% [-0.96%, -0.22%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 61.4% [61.2%, 61.6%] | 32.1% [31.6%, 32.7%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 64.1% [63.8%, 64.4%] | 39.7% [39.1%, 40.5%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 71.0% [70.9%, 71.1%] | 55.4% [55.2%, 55.6%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 74.9% [74.8%, 74.9%] | 68.5% [68.3%, 68.7%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 57.3% [57.2%, 57.6%] | 21.3% [21.0%, 22.0%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 62.0% [61.9%, 62.2%] | 32.5% [32.1%, 33.1%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 64.8% [64.6%, 65.3%] | 41.4% [40.8%, 42.5%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 72.4% [72.4%, 72.5%] | 60.3% [60.1%, 60.5%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 63.4% [63.4%, 63.4%] | 38.1% [38.1%, 38.1%] |

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
| 0.5% | 57.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 62.0% | 0.14% [0.00%, 0.42%] | -0.00% [-0.08%, 0.08%] | 0.00 |
| 2% | 64.8% | 1.96% [0.65%, 3.64%] | 0.32% [0.07%, 0.66%] | 0.00 |
| 5% | 72.4% | 0.30% [-0.36%, 1.15%] | -0.05% [-0.25%, 0.16%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
