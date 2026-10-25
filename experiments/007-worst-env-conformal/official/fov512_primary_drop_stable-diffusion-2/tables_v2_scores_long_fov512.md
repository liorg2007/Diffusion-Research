# H2-v2 tables (scores_long_fov512)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 15.9% [14.7%, 18.4%] | 11.3% [10.4%, 13.3%] |
| pooled@0.01 | 1.68% [0.00%, 5.04%] | 1.68% [0.00%, 5.04%] | 20.9% [19.7%, 23.2%] | 15.3% [14.4%, 17.3%] |
| pooled@0.02 | 2.29% [0.00%, 6.52%] | 2.51% [0.00%, 7.26%] | 41.4% [40.4%, 43.3%] | 31.8% [30.7%, 34.1%] |
| pooled@0.05 | 5.25% [1.35%, 10.76%] | 5.59% [1.69%, 11.67%] | 75.1% [73.8%, 76.6%] | 73.8% [72.9%, 74.8%] |
| wg@0.005 | 0.89% [0.00%, 2.69%] | 1.12% [0.00%, 3.36%] | 14.0% [13.3%, 15.3%] | 9.8% [9.3%, 10.8%] |
| wg@0.01 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 17.2% [16.0%, 19.7%] | 12.6% [11.6%, 14.6%] |
| wg@0.02 | 2.23% [0.00%, 6.35%] | 2.51% [0.00%, 7.26%] | 38.2% [37.4%, 40.0%] | 29.2% [28.3%, 31.2%] |
| wg@0.05 | 5.03% [1.24%, 10.45%] | 5.31% [1.68%, 11.26%] | 72.8% [71.4%, 74.4%] | 71.2% [69.9%, 72.5%] |
| M1 | 3.85% [0.73%, 8.74%] | 4.19% [0.84%, 9.39%] | 63.9% [63.9%, 63.9%] | 60.6% [60.6%, 60.6%] |

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
| 0.5% | 14.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 17.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 38.2% | 0.01% [0.00%, 0.03%] | 0.01% [0.00%, 0.03%] | 0.00 |
| 5% | 72.8% | 0.04% [-0.27%, 0.58%] | 0.07% [-0.08%, 0.21%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.51% [0.00%, 4.54%] | 1.68% [0.00%, 5.04%] | 12.5% [11.5%, 14.6%] | 11.3% [10.5%, 12.8%] |
| pooled@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 18.5% [17.3%, 20.9%] | 14.5% [13.3%, 16.9%] |
| pooled@0.02 | 2.35% [0.00%, 6.69%] | 2.79% [0.00%, 7.80%] | 29.8% [27.8%, 33.1%] | 23.9% [21.8%, 27.4%] |
| pooled@0.05 | 4.97% [1.35%, 10.03%] | 5.31% [1.69%, 10.83%] | 74.5% [73.9%, 75.3%] | 69.8% [69.2%, 70.6%] |
| wg@0.005 | 0.84% [0.00%, 2.52%] | 1.12% [0.00%, 3.36%] | 9.8% [9.4%, 10.7%] | 9.6% [9.3%, 10.0%] |
| wg@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 15.1% [13.8%, 17.8%] | 12.8% [11.5%, 15.2%] |
| wg@0.02 | 2.01% [0.00%, 5.85%] | 2.23% [0.00%, 6.41%] | 24.1% [22.7%, 26.8%] | 18.2% [16.7%, 21.0%] |
| wg@0.05 | 4.86% [1.28%, 9.92%] | 5.03% [1.68%, 10.56%] | 73.9% [73.4%, 74.7%] | 69.4% [68.8%, 70.0%] |
| M1 | 3.69% [0.67%, 8.40%] | 4.19% [0.84%, 9.34%] | 61.8% [61.8%, 61.8%] | 59.1% [59.1%, 59.1%] |

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
| 0.5% | 9.8% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 15.1% | 0.23% [0.00%, 0.52%] | 0.23% [0.00%, 0.52%] | 0.00 |
| 2% | 24.1% | 0.00% [0.00%, 0.00%] | 0.03% [0.00%, 0.06%] | 0.00 |
| 5% | 73.9% | -0.22% [-0.72%, 0.11%] | -0.06% [-0.28%, 0.07%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.73% [0.22%, 1.34%] | 1.40% [0.56%, 2.53%] | 38.2% [37.9%, 38.7%] | 7.7% [7.4%, 8.2%] |
| pooled@0.01 | 1.68% [0.89%, 2.57%] | 3.63% [1.93%, 5.87%] | 47.2% [47.1%, 47.4%] | 18.2% [18.1%, 18.4%] |
| pooled@0.02 | 3.24% [1.84%, 4.85%] | 5.03% [3.06%, 7.56%] | 52.0% [51.9%, 52.1%] | 23.8% [23.5%, 24.2%] |
| pooled@0.05 | 7.65% [4.97%, 10.90%] | 13.97% [10.19%, 18.13%] | 59.8% [59.5%, 60.0%] | 37.2% [36.7%, 37.9%] |
| wg@0.005 | 0.22% [0.06%, 0.45%] | 0.56% [0.28%, 1.40%] | 34.2% [34.1%, 34.5%] | 5.4% [5.3%, 5.5%] |
| wg@0.01 | 0.95% [0.28%, 1.72%] | 1.68% [0.56%, 3.09%] | 40.9% [40.1%, 41.7%] | 10.9% [10.0%, 11.8%] |
| wg@0.02 | 2.35% [1.18%, 3.71%] | 3.63% [1.95%, 5.88%] | 50.5% [50.5%, 50.6%] | 20.8% [20.7%, 20.9%] |
| wg@0.05 | 6.20% [3.87%, 9.08%] | 10.34% [6.82%, 14.25%] | 57.1% [56.7%, 57.5%] | 33.7% [33.3%, 34.2%] |
| M1 | 6.59% [4.33%, 9.44%] | 12.01% [8.54%, 16.01%] | 58.4% [58.4%, 58.4%] | 34.7% [34.7%, 34.7%] |

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
| 0.5% | 34.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 40.9% | -0.15% [-0.73%, 0.45%] | 0.03% [-0.14%, 0.22%] | 0.00 |
| 2% | 50.5% | 0.00% [-0.68%, 0.05%] | -0.16% [-0.36%, 0.02%] | 0.00 |
| 5% | 57.1% | -0.43% [-1.58%, 0.59%] | 0.13% [-0.12%, 0.40%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.78%, 2.49%] | 3.91% [1.94%, 7.28%] | 13.1% [12.8%, 13.5%] | 10.8% [10.6%, 11.1%] |
| pooled@0.01 | 2.63% [1.50%, 4.01%] | 6.15% [3.37%, 10.73%] | 20.4% [20.0%, 21.0%] | 18.1% [17.5%, 18.8%] |
| pooled@0.02 | 5.25% [3.33%, 7.37%] | 10.89% [7.24%, 16.57%] | 30.8% [30.4%, 31.4%] | 31.6% [31.0%, 32.2%] |
| pooled@0.05 | 11.45% [8.20%, 15.06%] | 21.51% [16.80%, 28.29%] | 45.8% [45.4%, 46.3%] | 48.4% [48.0%, 49.0%] |
| wg@0.005 | 0.61% [0.17%, 1.12%] | 1.40% [0.56%, 3.62%] | 7.5% [7.4%, 7.6%] | 4.7% [4.6%, 4.8%] |
| wg@0.01 | 1.79% [0.89%, 2.82%] | 4.19% [1.96%, 7.54%] | 14.7% [14.7%, 14.8%] | 12.0% [11.9%, 12.0%] |
| wg@0.02 | 2.91% [1.58%, 4.48%] | 6.70% [3.63%, 11.76%] | 22.6% [21.9%, 23.4%] | 20.5% [19.6%, 21.7%] |
| wg@0.05 | 9.22% [6.37%, 12.45%] | 16.76% [13.26%, 23.60%] | 40.5% [40.0%, 41.2%] | 43.4% [42.9%, 44.1%] |
| M1 | 0.06% [0.00%, 0.17%] | 0.28% [0.00%, 0.84%] | 0.6% [0.6%, 0.6%] | 1.4% [1.4%, 1.4%] |

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
| 0.5% | 7.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 14.7% | 0.02% [-0.05%, 0.10%] | 0.02% [-0.02%, 0.08%] | 0.00 |
| 2% | 22.6% | -0.56% [-1.44%, 0.23%] | -0.25% [-0.55%, 0.00%] | 0.00 |
| 5% | 40.5% | -0.00% [-0.99%, 0.90%] | 0.01% [-0.28%, 0.29%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.95% [0.17%, 1.90%] | 1.68% [0.56%, 3.90%] | 77.1% [76.9%, 77.4%] | 12.6% [12.6%, 12.7%] |
| pooled@0.01 | 1.51% [0.45%, 2.82%] | 2.51% [1.11%, 4.78%] | 80.3% [80.0%, 80.6%] | 13.9% [13.7%, 14.2%] |
| pooled@0.02 | 2.46% [1.12%, 4.07%] | 3.35% [1.95%, 6.16%] | 86.8% [86.6%, 87.1%] | 21.5% [21.2%, 21.8%] |
| pooled@0.05 | 7.09% [4.82%, 9.50%] | 10.34% [7.87%, 14.09%] | 94.7% [94.6%, 94.8%] | 35.4% [35.2%, 35.7%] |
| wg@0.005 | 0.17% [0.00%, 0.45%] | 0.28% [0.00%, 1.12%] | 59.7% [59.2%, 60.5%] | 6.0% [5.8%, 6.3%] |
| wg@0.01 | 1.28% [0.39%, 2.46%] | 1.96% [0.84%, 4.16%] | 79.6% [79.5%, 79.7%] | 13.2% [13.1%, 13.3%] |
| wg@0.02 | 1.90% [0.68%, 3.41%] | 2.79% [1.13%, 5.07%] | 83.9% [83.4%, 84.4%] | 17.3% [16.8%, 17.9%] |
| wg@0.05 | 6.26% [4.26%, 8.30%] | 8.94% [6.92%, 12.29%] | 93.8% [93.7%, 94.0%] | 33.2% [32.9%, 33.6%] |
| M1 | 5.87% [4.20%, 7.65%] | 8.66% [6.80%, 12.32%] | 94.0% [94.0%, 94.0%] | 33.2% [33.2%, 33.2%] |

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
| 0.5% | 59.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 79.6% | -0.23% [-0.63%, 0.03%] | -0.03% [-0.11%, 0.03%] | 0.00 |
| 2% | 83.9% | 0.00% [-0.72%, 0.13%] | 0.02% [-0.08%, 0.12%] | 0.00 |
| 5% | 93.8% | -0.34% [-1.32%, 0.39%] | 0.03% [-0.18%, 0.28%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.73% [0.73%, 2.96%] | 5.03% [1.94%, 9.09%] | 25.0% [24.8%, 25.2%] | 14.4% [14.2%, 14.7%] |
| pooled@0.01 | 3.41% [1.90%, 5.20%] | 6.70% [4.18%, 10.31%] | 32.9% [32.7%, 33.2%] | 24.4% [24.0%, 24.8%] |
| pooled@0.02 | 7.04% [4.51%, 10.08%] | 14.80% [10.61%, 20.22%] | 42.5% [42.0%, 43.1%] | 39.3% [38.8%, 39.9%] |
| pooled@0.05 | 14.58% [10.75%, 18.61%] | 37.99% [30.47%, 45.51%] | 55.9% [55.4%, 56.5%] | 60.4% [59.7%, 61.3%] |
| wg@0.005 | 0.84% [0.23%, 1.61%] | 3.63% [1.12%, 6.93%] | 18.3% [18.2%, 18.4%] | 9.1% [9.0%, 9.2%] |
| wg@0.01 | 1.34% [0.45%, 2.41%] | 5.03% [1.68%, 9.09%] | 22.8% [22.5%, 23.1%] | 12.7% [12.6%, 13.0%] |
| wg@0.02 | 2.46% [1.06%, 4.16%] | 7.54% [2.82%, 13.13%] | 29.5% [29.2%, 30.0%] | 20.1% [19.7%, 20.8%] |
| wg@0.05 | 7.77% [5.03%, 11.08%] | 15.08% [10.96%, 22.63%] | 46.3% [45.6%, 47.1%] | 44.5% [43.5%, 45.7%] |
| M1 | 10.56% [7.78%, 13.56%] | 29.61% [23.31%, 35.96%] | 49.2% [49.2%, 49.2%] | 50.6% [50.6%, 50.6%] |

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
| 0.5% | 18.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 22.8% | 0.12% [0.00%, 0.38%] | -0.04% [-0.14%, 0.05%] | 0.00 |
| 2% | 29.5% | 1.92% [-1.69%, 4.53%] | -0.29% [-0.78%, 0.17%] | 0.00 |
| 5% | 46.3% | -6.85% [-8.86%, -0.65%] | -1.34% [-2.04%, -0.80%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)
