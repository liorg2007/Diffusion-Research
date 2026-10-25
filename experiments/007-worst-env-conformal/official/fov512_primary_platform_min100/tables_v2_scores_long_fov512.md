# H2-v2 tables (scores_long_fov512)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 16.8% [15.6%, 19.4%] | 13.1% [12.2%, 15.0%] |
| pooled@0.01 | 1.68% [0.00%, 5.04%] | 1.68% [0.00%, 5.04%] | 22.2% [21.0%, 24.5%] | 17.0% [16.1%, 19.0%] |
| pooled@0.02 | 2.29% [0.00%, 6.52%] | 2.51% [0.00%, 7.26%] | 43.0% [42.0%, 44.8%] | 32.8% [31.6%, 35.0%] |
| pooled@0.05 | 5.25% [1.35%, 10.76%] | 5.59% [1.69%, 11.67%] | 76.6% [75.4%, 78.0%] | 73.6% [72.5%, 74.8%] |
| wg@0.005 | 0.89% [0.00%, 2.69%] | 1.12% [0.00%, 3.36%] | 14.5% [13.8%, 15.9%] | 11.4% [10.9%, 12.4%] |
| wg@0.01 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 18.3% [17.0%, 20.8%] | 14.3% [13.4%, 16.2%] |
| wg@0.02 | 2.23% [0.00%, 6.35%] | 2.51% [0.00%, 7.26%] | 39.9% [39.0%, 41.5%] | 30.2% [29.2%, 32.1%] |
| wg@0.05 | 5.03% [1.24%, 10.45%] | 5.31% [1.68%, 11.26%] | 74.5% [73.1%, 76.0%] | 71.0% [69.6%, 72.5%] |
| M1 | 3.85% [0.73%, 8.74%] | 4.19% [0.84%, 9.39%] | 65.5% [65.5%, 65.5%] | 60.5% [60.5%, 60.5%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 4.19% | 3.91% | 4.19% | 3.63% | 3.35% |
| pooled@0.01 | 1.68% | 1.68% | 1.68% | 1.68% | 1.68% |
| wg@0.01 | 1.68% | 1.68% | 1.68% | 1.40% | 1.40% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 14.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 18.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 39.9% | 0.01% [0.00%, 0.03%] | 0.01% [0.00%, 0.03%] | 0.00 |
| 5% | 74.5% | 0.03% [-0.28%, 0.57%] | 0.07% [-0.08%, 0.20%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.51% [0.00%, 4.54%] | 1.68% [0.00%, 5.04%] | 15.3% [14.3%, 17.4%] | 13.3% [12.5%, 14.9%] |
| pooled@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 22.1% [20.9%, 24.4%] | 16.8% [15.6%, 19.2%] |
| pooled@0.02 | 2.35% [0.00%, 6.69%] | 2.79% [0.00%, 7.80%] | 33.5% [31.6%, 36.6%] | 26.4% [24.3%, 29.9%] |
| pooled@0.05 | 4.97% [1.35%, 10.03%] | 5.31% [1.69%, 10.83%] | 75.7% [75.1%, 76.5%] | 71.7% [71.1%, 72.4%] |
| wg@0.005 | 0.84% [0.00%, 2.52%] | 1.12% [0.00%, 3.36%] | 12.6% [12.2%, 13.6%] | 11.3% [11.1%, 11.9%] |
| wg@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 18.0% [16.7%, 20.7%] | 15.0% [13.7%, 17.5%] |
| wg@0.02 | 2.01% [0.00%, 5.85%] | 2.23% [0.00%, 6.41%] | 27.7% [26.3%, 30.3%] | 20.5% [19.0%, 23.4%] |
| wg@0.05 | 4.86% [1.28%, 9.92%] | 5.03% [1.68%, 10.56%] | 75.2% [74.7%, 75.9%] | 71.2% [70.7%, 71.8%] |
| M1 | 3.69% [0.67%, 8.40%] | 4.19% [0.84%, 9.34%] | 64.5% [64.5%, 64.5%] | 60.7% [60.7%, 60.7%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 3.91% | 4.19% | 4.19% | 3.07% | 3.07% |
| pooled@0.01 | 1.96% | 1.96% | 1.96% | 1.40% | 1.68% |
| wg@0.01 | 1.96% | 1.96% | 1.96% | 1.40% | 1.68% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 12.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 18.0% | 0.24% [0.00%, 0.57%] | 0.24% [0.00%, 0.57%] | 0.00 |
| 2% | 27.7% | 0.00% [0.00%, 0.00%] | 0.03% [0.00%, 0.07%] | 0.00 |
| 5% | 75.2% | -0.22% [-0.71%, 0.12%] | -0.06% [-0.28%, 0.08%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.73% [0.22%, 1.34%] | 1.40% [0.56%, 2.53%] | 44.0% [43.7%, 44.4%] | 7.3% [7.1%, 7.7%] |
| pooled@0.01 | 1.68% [0.89%, 2.57%] | 3.63% [1.93%, 5.87%] | 52.7% [52.6%, 52.9%] | 16.7% [16.6%, 16.9%] |
| pooled@0.02 | 3.24% [1.84%, 4.85%] | 5.03% [3.06%, 7.56%] | 57.0% [56.9%, 57.1%] | 22.6% [22.3%, 23.0%] |
| pooled@0.05 | 7.65% [4.97%, 10.90%] | 13.97% [10.19%, 18.13%] | 64.2% [64.0%, 64.4%] | 36.5% [36.0%, 37.2%] |
| wg@0.005 | 0.22% [0.06%, 0.45%] | 0.56% [0.28%, 1.40%] | 39.2% [39.0%, 39.4%] | 5.0% [4.9%, 5.1%] |
| wg@0.01 | 0.95% [0.28%, 1.72%] | 1.68% [0.56%, 3.09%] | 46.6% [45.9%, 47.4%] | 10.2% [9.4%, 11.0%] |
| wg@0.02 | 2.35% [1.18%, 3.71%] | 3.63% [1.95%, 5.88%] | 55.7% [55.6%, 55.7%] | 19.5% [19.3%, 19.6%] |
| wg@0.05 | 6.20% [3.87%, 9.08%] | 10.34% [6.82%, 14.25%] | 61.7% [61.3%, 62.2%] | 32.8% [32.4%, 33.2%] |
| M1 | 6.59% [4.33%, 9.44%] | 12.01% [8.54%, 16.01%] | 63.0% [63.0%, 63.0%] | 33.5% [33.5%, 33.5%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 3.63% | 5.59% | 4.19% | 12.01% | 7.54% |
| pooled@0.01 | 0.84% | 1.12% | 0.56% | 3.63% | 2.23% |
| wg@0.01 | 0.56% | 1.12% | 0.28% | 1.68% | 1.12% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 39.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 46.6% | -0.15% [-0.73%, 0.45%] | 0.03% [-0.14%, 0.22%] | 0.00 |
| 2% | 55.7% | 0.00% [-0.68%, 0.05%] | -0.16% [-0.36%, 0.02%] | 0.00 |
| 5% | 61.7% | -0.45% [-1.62%, 0.56%] | 0.12% [-0.13%, 0.39%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.78%, 2.49%] | 3.91% [1.94%, 7.28%] | 14.3% [14.1%, 14.7%] | 11.0% [10.8%, 11.4%] |
| pooled@0.01 | 2.63% [1.50%, 4.01%] | 6.15% [3.37%, 10.73%] | 21.8% [21.4%, 22.4%] | 18.7% [18.1%, 19.5%] |
| pooled@0.02 | 5.25% [3.33%, 7.37%] | 10.89% [7.24%, 16.57%] | 32.0% [31.5%, 32.5%] | 32.5% [31.9%, 33.2%] |
| pooled@0.05 | 11.45% [8.20%, 15.06%] | 21.51% [16.80%, 28.29%] | 47.3% [46.9%, 47.8%] | 49.9% [49.5%, 50.5%] |
| wg@0.005 | 0.61% [0.17%, 1.12%] | 1.40% [0.56%, 3.62%] | 8.4% [8.4%, 8.5%] | 4.9% [4.8%, 5.1%] |
| wg@0.01 | 1.79% [0.89%, 2.82%] | 4.19% [1.96%, 7.54%] | 15.9% [15.8%, 16.0%] | 12.3% [12.2%, 12.4%] |
| wg@0.02 | 2.91% [1.58%, 4.48%] | 6.70% [3.63%, 11.76%] | 23.9% [23.3%, 24.8%] | 21.2% [20.2%, 22.5%] |
| wg@0.05 | 9.22% [6.37%, 12.45%] | 16.76% [13.26%, 23.60%] | 41.8% [41.3%, 42.5%] | 44.8% [44.3%, 45.5%] |
| M1 | 0.06% [0.00%, 0.17%] | 0.28% [0.00%, 0.84%] | 1.1% [1.1%, 1.1%] | 1.5% [1.5%, 1.5%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.28% |
| pooled@0.01 | 0.28% | 1.12% | 1.68% | 6.15% | 3.91% |
| wg@0.01 | 0.28% | 0.56% | 1.40% | 4.19% | 2.51% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 8.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 15.9% | 0.02% [-0.04%, 0.11%] | 0.02% [-0.02%, 0.09%] | 0.00 |
| 2% | 23.9% | -0.53% [-1.41%, 0.25%] | -0.24% [-0.53%, 0.01%] | 0.00 |
| 5% | 41.8% | 0.07% [-0.92%, 1.00%] | 0.05% [-0.24%, 0.32%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.95% [0.17%, 1.90%] | 1.68% [0.56%, 3.90%] | 80.0% [79.8%, 80.2%] | 15.1% [15.1%, 15.2%] |
| pooled@0.01 | 1.51% [0.45%, 2.82%] | 2.51% [1.11%, 4.78%] | 82.7% [82.5%, 83.0%] | 16.4% [16.2%, 16.6%] |
| pooled@0.02 | 2.46% [1.12%, 4.07%] | 3.35% [1.95%, 6.16%] | 88.5% [88.2%, 88.7%] | 24.1% [23.9%, 24.4%] |
| pooled@0.05 | 7.09% [4.82%, 9.50%] | 10.34% [7.87%, 14.09%] | 95.3% [95.3%, 95.4%] | 39.5% [39.2%, 39.8%] |
| wg@0.005 | 0.17% [0.00%, 0.45%] | 0.28% [0.00%, 1.12%] | 63.0% [62.6%, 63.8%] | 7.0% [6.8%, 7.4%] |
| wg@0.01 | 1.28% [0.39%, 2.46%] | 1.96% [0.84%, 4.16%] | 82.1% [82.1%, 82.2%] | 15.7% [15.7%, 15.8%] |
| wg@0.02 | 1.90% [0.68%, 3.41%] | 2.79% [1.13%, 5.07%] | 85.9% [85.5%, 86.3%] | 19.7% [19.2%, 20.3%] |
| wg@0.05 | 6.26% [4.26%, 8.30%] | 8.94% [6.92%, 12.29%] | 94.6% [94.5%, 94.7%] | 37.1% [36.7%, 37.5%] |
| M1 | 5.87% [4.20%, 7.65%] | 8.66% [6.80%, 12.32%] | 94.8% [94.8%, 94.8%] | 37.2% [37.2%, 37.2%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 5.03% | 3.91% | 3.63% | 8.10% | 8.66% |
| pooled@0.01 | 1.40% | 0.56% | 1.12% | 1.96% | 2.51% |
| wg@0.01 | 1.40% | 0.56% | 0.84% | 1.68% | 1.96% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 63.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 82.1% | -0.23% [-0.63%, 0.03%] | -0.03% [-0.11%, 0.03%] | 0.00 |
| 2% | 85.9% | 0.00% [-0.72%, 0.13%] | 0.02% [-0.08%, 0.12%] | 0.00 |
| 5% | 94.6% | -0.34% [-1.32%, 0.39%] | 0.03% [-0.18%, 0.28%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.73% [0.73%, 2.96%] | 5.03% [1.94%, 9.09%] | 26.4% [26.2%, 26.6%] | 16.2% [16.1%, 16.4%] |
| pooled@0.01 | 3.41% [1.90%, 5.20%] | 6.70% [4.18%, 10.31%] | 34.2% [34.0%, 34.5%] | 26.3% [25.9%, 26.8%] |
| pooled@0.02 | 7.04% [4.51%, 10.08%] | 14.80% [10.61%, 20.22%] | 44.2% [43.7%, 44.8%] | 41.7% [41.2%, 42.4%] |
| pooled@0.05 | 14.58% [10.75%, 18.61%] | 37.99% [30.47%, 45.51%] | 58.3% [57.8%, 58.8%] | 63.2% [62.5%, 64.0%] |
| wg@0.005 | 0.84% [0.23%, 1.61%] | 3.63% [1.12%, 6.93%] | 18.4% [18.3%, 18.5%] | 10.6% [10.6%, 10.7%] |
| wg@0.01 | 1.34% [0.45%, 2.41%] | 5.03% [1.68%, 9.09%] | 24.2% [23.9%, 24.5%] | 14.6% [14.5%, 14.8%] |
| wg@0.02 | 2.46% [1.06%, 4.16%] | 7.54% [2.82%, 13.13%] | 30.8% [30.5%, 31.3%] | 21.8% [21.4%, 22.5%] |
| wg@0.05 | 7.77% [5.03%, 11.08%] | 15.08% [10.96%, 22.63%] | 48.0% [47.3%, 48.9%] | 47.0% [46.0%, 48.3%] |
| M1 | 10.56% [7.78%, 13.56%] | 29.61% [23.31%, 35.96%] | 52.0% [52.0%, 52.0%] | 53.0% [53.0%, 53.0%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 7.82% | 1.68% | 0.84% | 29.61% | 12.85% |
| pooled@0.01 | 5.31% | 0.00% | 0.00% | 6.70% | 5.03% |
| wg@0.01 | 5.03% | 0.00% | 0.00% | 0.28% | 1.40% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 18.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 24.2% | 0.10% [0.00%, 0.33%] | -0.05% [-0.16%, 0.04%] | 0.00 |
| 2% | 30.8% | 1.95% [-1.52%, 4.54%] | -0.24% [-0.71%, 0.21%] | 0.00 |
| 5% | 48.0% | -6.45% [-8.35%, -0.39%] | -1.22% [-1.92%, -0.69%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)
