# H2-v2 tables (scores_long_fov512)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 16.0% [14.8%, 18.5%] | 12.5% [11.6%, 14.3%] |
| pooled@0.01 | 1.68% [0.00%, 5.04%] | 1.68% [0.00%, 5.04%] | 21.3% [20.1%, 23.5%] | 16.5% [15.5%, 18.3%] |
| pooled@0.02 | 2.29% [0.00%, 6.52%] | 2.51% [0.00%, 7.26%] | 41.4% [40.4%, 43.3%] | 31.4% [30.3%, 33.5%] |
| pooled@0.05 | 5.25% [1.35%, 10.76%] | 5.59% [1.69%, 11.67%] | 75.0% [73.7%, 76.4%] | 71.4% [70.2%, 72.6%] |
| wg@0.005 | 0.89% [0.00%, 2.69%] | 1.12% [0.00%, 3.36%] | 13.6% [12.9%, 14.9%] | 10.6% [10.1%, 11.6%] |
| wg@0.01 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 17.5% [16.2%, 19.9%] | 13.6% [12.6%, 15.4%] |
| wg@0.02 | 2.23% [0.00%, 6.35%] | 2.51% [0.00%, 7.26%] | 38.1% [37.2%, 39.8%] | 29.2% [28.3%, 31.1%] |
| wg@0.05 | 5.03% [1.24%, 10.45%] | 5.31% [1.68%, 11.26%] | 73.0% [71.7%, 74.5%] | 68.5% [67.1%, 70.1%] |
| M1 | 3.85% [0.73%, 8.74%] | 4.19% [0.84%, 9.39%] | 64.4% [64.4%, 64.4%] | 58.2% [58.2%, 58.2%] |

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
| 0.5% | 13.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 17.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 38.1% | 0.01% [0.00%, 0.03%] | 0.01% [0.00%, 0.03%] | 0.00 |
| 5% | 73.0% | 0.02% [-0.30%, 0.55%] | 0.05% [-0.10%, 0.19%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.51% [0.00%, 4.54%] | 1.68% [0.00%, 5.04%] | 14.5% [13.6%, 16.5%] | 12.6% [11.9%, 14.0%] |
| pooled@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 20.9% [19.7%, 23.2%] | 15.5% [14.4%, 17.8%] |
| pooled@0.02 | 2.35% [0.00%, 6.69%] | 2.79% [0.00%, 7.80%] | 31.8% [30.0%, 34.8%] | 24.6% [22.6%, 27.9%] |
| pooled@0.05 | 4.97% [1.35%, 10.03%] | 5.31% [1.69%, 10.83%] | 73.3% [72.5%, 74.1%] | 69.2% [68.6%, 69.9%] |
| wg@0.005 | 0.84% [0.00%, 2.52%] | 1.12% [0.00%, 3.36%] | 11.7% [11.3%, 12.6%] | 10.8% [10.5%, 11.2%] |
| wg@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 17.1% [15.8%, 19.7%] | 14.1% [13.0%, 16.4%] |
| wg@0.02 | 2.01% [0.00%, 5.85%] | 2.23% [0.00%, 6.41%] | 26.2% [24.8%, 28.7%] | 19.3% [17.9%, 22.0%] |
| wg@0.05 | 4.86% [1.28%, 9.92%] | 5.03% [1.68%, 10.56%] | 72.8% [72.2%, 73.6%] | 68.6% [68.1%, 69.2%] |
| M1 | 3.69% [0.67%, 8.40%] | 4.19% [0.84%, 9.34%] | 61.6% [61.6%, 61.6%] | 58.6% [58.6%, 58.6%] |

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
| 0.5% | 11.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 17.1% | 0.24% [0.00%, 0.56%] | 0.24% [0.00%, 0.56%] | 0.00 |
| 2% | 26.2% | 0.00% [0.00%, 0.00%] | 0.04% [0.00%, 0.09%] | 0.00 |
| 5% | 72.8% | -0.23% [-0.72%, 0.10%] | -0.07% [-0.29%, 0.07%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.73% [0.22%, 1.34%] | 1.40% [0.56%, 2.53%] | 38.6% [38.3%, 39.1%] | 6.0% [5.7%, 6.3%] |
| pooled@0.01 | 1.68% [0.89%, 2.57%] | 3.63% [1.93%, 5.87%] | 47.7% [47.6%, 47.8%] | 13.7% [13.6%, 13.7%] |
| pooled@0.02 | 3.24% [1.84%, 4.85%] | 5.03% [3.06%, 7.56%] | 51.8% [51.7%, 51.9%] | 19.2% [18.8%, 19.6%] |
| pooled@0.05 | 7.65% [4.97%, 10.90%] | 13.97% [10.19%, 18.13%] | 59.8% [59.5%, 60.0%] | 32.9% [32.4%, 33.6%] |
| wg@0.005 | 0.22% [0.06%, 0.45%] | 0.56% [0.28%, 1.40%] | 34.1% [34.0%, 34.4%] | 3.7% [3.6%, 3.8%] |
| wg@0.01 | 0.95% [0.28%, 1.72%] | 1.68% [0.56%, 3.09%] | 41.4% [40.6%, 42.2%] | 8.4% [7.7%, 9.1%] |
| wg@0.02 | 2.35% [1.18%, 3.71%] | 3.63% [1.95%, 5.88%] | 50.3% [50.2%, 50.3%] | 16.0% [15.9%, 16.2%] |
| wg@0.05 | 6.20% [3.87%, 9.08%] | 10.34% [6.82%, 14.25%] | 57.1% [56.7%, 57.6%] | 29.3% [28.9%, 29.8%] |
| M1 | 6.59% [4.33%, 9.44%] | 12.01% [8.54%, 16.01%] | 58.4% [58.4%, 58.4%] | 30.0% [30.0%, 30.0%] |

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
| 0.5% | 34.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 41.4% | -0.15% [-0.73%, 0.45%] | 0.03% [-0.14%, 0.22%] | 0.00 |
| 2% | 50.3% | 0.00% [-0.63%, 0.06%] | -0.14% [-0.31%, 0.04%] | 0.00 |
| 5% | 57.1% | -0.48% [-1.63%, 0.52%] | 0.11% [-0.15%, 0.38%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.78%, 2.49%] | 3.91% [1.94%, 7.28%] | 14.5% [14.3%, 14.8%] | 10.4% [10.1%, 10.8%] |
| pooled@0.01 | 2.63% [1.50%, 4.01%] | 6.15% [3.37%, 10.73%] | 21.6% [21.1%, 22.2%] | 17.9% [17.3%, 18.6%] |
| pooled@0.02 | 5.25% [3.33%, 7.37%] | 10.89% [7.24%, 16.57%] | 32.0% [31.5%, 32.6%] | 30.5% [29.9%, 31.1%] |
| pooled@0.05 | 11.45% [8.20%, 15.06%] | 21.51% [16.80%, 28.29%] | 47.4% [47.0%, 47.9%] | 46.6% [46.2%, 47.2%] |
| wg@0.005 | 0.61% [0.17%, 1.12%] | 1.40% [0.56%, 3.62%] | 8.6% [8.5%, 8.7%] | 4.8% [4.8%, 5.0%] |
| wg@0.01 | 1.79% [0.89%, 2.82%] | 4.19% [1.96%, 7.54%] | 16.0% [15.9%, 16.1%] | 11.8% [11.7%, 11.9%] |
| wg@0.02 | 2.91% [1.58%, 4.48%] | 6.70% [3.63%, 11.76%] | 23.6% [22.9%, 24.5%] | 20.2% [19.3%, 21.4%] |
| wg@0.05 | 9.22% [6.37%, 12.45%] | 16.76% [13.26%, 23.60%] | 41.7% [41.2%, 42.5%] | 41.4% [40.9%, 42.2%] |
| M1 | 0.06% [0.00%, 0.17%] | 0.28% [0.00%, 0.84%] | 1.3% [1.3%, 1.3%] | 1.4% [1.4%, 1.4%] |

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
| 0.5% | 8.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 16.0% | 0.02% [-0.06%, 0.10%] | 0.02% [-0.02%, 0.08%] | 0.00 |
| 2% | 23.6% | -0.48% [-1.32%, 0.28%] | -0.22% [-0.49%, 0.03%] | 0.00 |
| 5% | 41.7% | 0.19% [-0.81%, 1.14%] | 0.12% [-0.18%, 0.40%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.95% [0.17%, 1.90%] | 1.68% [0.56%, 3.90%] | 77.1% [76.9%, 77.4%] | 11.8% [11.8%, 11.8%] |
| pooled@0.01 | 1.51% [0.45%, 2.82%] | 2.51% [1.11%, 4.78%] | 80.3% [80.0%, 80.6%] | 12.8% [12.6%, 13.0%] |
| pooled@0.02 | 2.46% [1.12%, 4.07%] | 3.35% [1.95%, 6.16%] | 86.8% [86.6%, 87.1%] | 19.7% [19.4%, 20.0%] |
| pooled@0.05 | 7.09% [4.82%, 9.50%] | 10.34% [7.87%, 14.09%] | 94.7% [94.6%, 94.8%] | 34.0% [33.7%, 34.2%] |
| wg@0.005 | 0.17% [0.00%, 0.45%] | 0.28% [0.00%, 1.12%] | 57.9% [57.5%, 58.8%] | 4.2% [4.0%, 4.5%] |
| wg@0.01 | 1.28% [0.39%, 2.46%] | 1.96% [0.84%, 4.16%] | 79.6% [79.5%, 79.7%] | 12.2% [12.2%, 12.3%] |
| wg@0.02 | 1.90% [0.68%, 3.41%] | 2.79% [1.13%, 5.07%] | 83.9% [83.4%, 84.4%] | 15.6% [15.1%, 16.0%] |
| wg@0.05 | 6.26% [4.26%, 8.30%] | 8.94% [6.92%, 12.29%] | 93.8% [93.7%, 94.0%] | 32.2% [31.9%, 32.5%] |
| M1 | 5.87% [4.20%, 7.65%] | 8.66% [6.80%, 12.32%] | 94.0% [94.0%, 94.0%] | 32.5% [32.5%, 32.5%] |

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
| 0.5% | 57.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 79.6% | -0.23% [-0.63%, 0.03%] | -0.03% [-0.11%, 0.03%] | 0.00 |
| 2% | 83.9% | 0.00% [-0.72%, 0.13%] | 0.02% [-0.08%, 0.12%] | 0.00 |
| 5% | 93.8% | -0.34% [-1.32%, 0.39%] | 0.03% [-0.18%, 0.28%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.73% [0.73%, 2.96%] | 5.03% [1.94%, 9.09%] | 21.4% [21.3%, 21.6%] | 14.3% [14.2%, 14.5%] |
| pooled@0.01 | 3.41% [1.90%, 5.20%] | 6.70% [4.18%, 10.31%] | 29.2% [29.0%, 29.5%] | 23.7% [23.3%, 24.2%] |
| pooled@0.02 | 7.04% [4.51%, 10.08%] | 14.80% [10.61%, 20.22%] | 39.3% [38.8%, 39.9%] | 39.2% [38.6%, 39.8%] |
| pooled@0.05 | 14.58% [10.75%, 18.61%] | 37.99% [30.47%, 45.51%] | 53.2% [52.7%, 53.8%] | 60.9% [60.1%, 61.7%] |
| wg@0.005 | 0.84% [0.23%, 1.61%] | 3.63% [1.12%, 6.93%] | 14.6% [14.5%, 14.7%] | 9.0% [9.0%, 9.1%] |
| wg@0.01 | 1.34% [0.45%, 2.41%] | 5.03% [1.68%, 9.09%] | 19.7% [19.4%, 19.9%] | 12.8% [12.7%, 13.0%] |
| wg@0.02 | 2.46% [1.06%, 4.16%] | 7.54% [2.82%, 13.13%] | 25.3% [24.9%, 25.8%] | 19.4% [18.9%, 20.1%] |
| wg@0.05 | 7.77% [5.03%, 11.08%] | 15.08% [10.96%, 22.63%] | 43.2% [42.4%, 44.1%] | 44.5% [43.5%, 45.7%] |
| M1 | 10.56% [7.78%, 13.56%] | 29.61% [23.31%, 35.96%] | 46.6% [46.6%, 46.6%] | 50.3% [50.3%, 50.3%] |

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
| 0.5% | 14.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 19.7% | 0.09% [0.00%, 0.30%] | -0.06% [-0.17%, 0.04%] | 0.00 |
| 2% | 25.3% | 2.03% [-1.20%, 4.57%] | -0.14% [-0.57%, 0.30%] | 0.00 |
| 5% | 43.2% | -6.58% [-8.49%, -0.50%] | -1.26% [-1.96%, -0.73%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)
