# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 30.9% [30.2%, 32.0%] | 23.0% [22.5%, 23.9%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 41.1% [40.4%, 42.3%] | 31.2% [30.5%, 32.3%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 52.7% [51.8%, 54.4%] | 42.1% [41.0%, 44.0%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 78.8% [78.5%, 79.3%] | 71.4% [71.0%, 72.0%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 18.3% [17.8%, 19.1%] | 10.3% [9.9%, 11.0%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 30.1% [29.4%, 31.3%] | 22.5% [21.9%, 23.5%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 42.9% [41.8%, 44.9%] | 33.3% [32.1%, 35.2%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 73.6% [72.8%, 74.4%] | 65.6% [64.7%, 66.7%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 47.2% [47.2%, 47.2%] | 38.8% [38.8%, 38.8%] |

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
| 0.5% | 18.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 30.1% | 0.09% [0.00%, 0.14%] | 0.11% [-0.08%, 0.25%] | 0.00 |
| 2% | 42.9% | 0.35% [0.00%, 0.98%] | 0.17% [0.00%, 0.39%] | 0.00 |
| 5% | 73.6% | -0.29% [-0.79%, 0.35%] | 0.00% [-0.17%, 0.17%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 27.4% [27.0%, 28.1%] | 16.8% [16.2%, 17.7%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 34.5% [33.1%, 37.0%] | 25.1% [23.7%, 27.8%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 55.0% [53.9%, 56.6%] | 46.1% [44.9%, 47.9%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 74.5% [74.2%, 75.0%] | 66.8% [66.4%, 67.3%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 19.5% [19.3%, 19.7%] | 8.9% [8.7%, 9.1%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 28.8% [28.0%, 30.4%] | 18.1% [17.2%, 19.7%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 40.9% [39.6%, 43.3%] | 32.1% [30.8%, 34.5%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 71.5% [71.2%, 72.0%] | 63.6% [63.3%, 64.1%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 61.0% [61.0%, 61.0%] | 51.5% [51.5%, 51.5%] |

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
| 0.5% | 19.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 28.8% | 0.30% [0.00%, 0.49%] | 0.17% [-0.17%, 0.48%] | 0.00 |
| 2% | 40.9% | 0.00% [-0.18%, 0.00%] | -0.07% [-0.21%, 0.07%] | 0.00 |
| 5% | 71.5% | 0.11% [-0.78%, 0.34%] | -0.23% [-0.49%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 58.9% [58.5%, 59.2%] | 12.9% [12.7%, 13.1%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 66.0% [65.7%, 66.3%] | 17.3% [17.2%, 17.6%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 77.2% [76.9%, 77.6%] | 27.2% [26.7%, 27.8%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 86.6% [86.4%, 86.8%] | 43.4% [43.2%, 43.6%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 29.5% [28.4%, 31.7%] | 5.2% [4.9%, 5.7%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 58.5% [58.3%, 58.8%] | 12.8% [12.7%, 12.9%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 63.9% [63.0%, 64.8%] | 17.3% [17.0%, 17.7%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 85.8% [85.7%, 86.1%] | 41.7% [41.3%, 42.1%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 85.2% [85.2%, 85.2%] | 38.4% [38.4%, 38.4%] |

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
| 0.5% | 29.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 58.5% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 63.9% | 0.20% [0.00%, 0.66%] | 0.03% [-0.04%, 0.13%] | 0.00 |
| 5% | 85.8% | -0.84% [-1.96%, 0.01%] | -0.23% [-0.50%, -0.00%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 11.7% [11.5%, 11.9%] | 14.3% [14.0%, 14.6%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 17.4% [17.2%, 17.5%] | 21.4% [21.2%, 21.6%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 23.3% [23.0%, 23.8%] | 29.1% [28.7%, 29.8%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 42.3% [42.1%, 42.6%] | 49.8% [49.5%, 50.1%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 6.0% [6.0%, 6.1%] | 6.3% [6.2%, 6.3%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 11.6% [11.4%, 12.0%] | 14.0% [13.7%, 14.6%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 17.9% [17.7%, 18.2%] | 21.9% [21.6%, 22.4%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 32.4% [32.0%, 33.0%] | 40.4% [39.9%, 41.1%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 29.0% [29.0%, 29.0%] | 36.3% [36.3%, 36.3%] |

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
| 0.5% | 6.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 11.6% | -2.59% [-4.68%, -1.01%] | -0.44% [-0.84%, -0.15%] | 0.00 |
| 2% | 17.9% | -2.17% [-3.34%, -1.05%] | -0.38% [-0.69%, -0.06%] | 0.00 |
| 5% | 32.4% | -2.02% [-3.48%, -0.74%] | -0.40% [-0.81%, -0.01%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 77.3% [76.7%, 78.5%] | 22.5% [21.7%, 24.0%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 88.1% [87.8%, 88.5%] | 35.5% [35.0%, 36.3%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 93.5% [93.4%, 93.7%] | 45.8% [45.4%, 46.3%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 97.9% [97.8%, 98.0%] | 60.0% [59.8%, 60.4%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 69.1% [68.7%, 69.9%] | 14.9% [14.5%, 15.7%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 74.8% [74.1%, 76.2%] | 20.2% [19.3%, 22.1%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 85.9% [85.2%, 86.8%] | 33.3% [32.4%, 34.6%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 96.4% [96.3%, 96.5%] | 53.2% [52.7%, 53.8%] |
| M1 | 2.29% [0.84%, 4.28%] | 3.91% [1.69%, 7.02%] | 92.0% [92.0%, 92.0%] | 43.5% [43.5%, 43.5%] |

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
| 0.5% | 69.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 74.8% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 85.9% | 2.35% [0.69%, 3.83%] | 0.53% [0.11%, 0.96%] | 0.00 |
| 5% | 96.4% | -1.40% [-2.54%, 0.17%] | -0.36% [-0.70%, -0.08%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 53.5% [53.3%, 53.8%] | 32.1% [31.6%, 32.7%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 56.8% [56.5%, 57.2%] | 39.7% [39.1%, 40.5%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 64.2% [64.1%, 64.4%] | 54.2% [54.0%, 54.5%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 69.6% [69.6%, 69.7%] | 67.4% [67.3%, 67.6%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 49.0% [48.8%, 49.3%] | 21.2% [20.8%, 21.9%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 54.3% [54.2%, 54.5%] | 32.5% [32.0%, 33.1%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 57.6% [57.3%, 58.1%] | 41.3% [40.7%, 42.3%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 66.2% [66.1%, 66.3%] | 59.1% [58.9%, 59.3%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 55.7% [55.7%, 55.7%] | 38.1% [38.1%, 38.1%] |

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
| 0.5% | 49.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 54.3% | 0.14% [0.00%, 0.44%] | 0.00% [-0.08%, 0.08%] | 0.00 |
| 2% | 57.6% | 2.03% [0.67%, 3.75%] | 0.34% [0.08%, 0.68%] | 0.00 |
| 5% | 66.2% | 0.38% [-0.28%, 1.28%] | 0.01% [-0.19%, 0.21%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
