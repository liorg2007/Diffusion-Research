# H2-v2 tables (scores_long_fov512)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 16.9% [15.7%, 19.5%] | 13.9% [12.9%, 15.8%] |
| pooled@0.01 | 1.68% [0.00%, 5.04%] | 1.68% [0.00%, 5.04%] | 22.4% [21.2%, 24.7%] | 17.4% [16.5%, 19.4%] |
| pooled@0.02 | 2.29% [0.00%, 6.52%] | 2.51% [0.00%, 7.26%] | 43.0% [42.0%, 44.9%] | 33.7% [32.6%, 35.8%] |
| pooled@0.05 | 5.25% [1.35%, 10.76%] | 5.59% [1.69%, 11.67%] | 76.5% [75.2%, 77.9%] | 74.0% [72.9%, 75.2%] |
| wg@0.005 | 0.89% [0.00%, 2.69%] | 1.12% [0.00%, 3.36%] | 14.3% [13.6%, 15.8%] | 12.2% [11.7%, 13.2%] |
| wg@0.01 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 18.0% [16.8%, 20.6%] | 14.9% [14.0%, 16.8%] |
| wg@0.02 | 2.23% [0.00%, 6.35%] | 2.51% [0.00%, 7.26%] | 39.5% [38.6%, 41.2%] | 30.7% [29.8%, 32.6%] |
| wg@0.05 | 5.03% [1.24%, 10.45%] | 5.31% [1.68%, 11.26%] | 74.4% [73.0%, 75.9%] | 71.3% [69.9%, 72.8%] |
| M1 | 3.85% [0.73%, 8.74%] | 4.19% [0.84%, 9.39%] | 65.6% [65.6%, 65.6%] | 60.9% [60.9%, 60.9%] |

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
| 0.5% | 14.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 18.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 39.5% | 0.01% [0.00%, 0.03%] | 0.01% [0.00%, 0.03%] | 0.00 |
| 5% | 74.4% | 0.05% [-0.26%, 0.59%] | 0.08% [-0.08%, 0.22%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.51% [0.00%, 4.54%] | 1.68% [0.00%, 5.04%] | 15.7% [14.7%, 17.7%] | 14.0% [13.2%, 15.5%] |
| pooled@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 22.4% [21.3%, 24.7%] | 17.3% [16.0%, 19.7%] |
| pooled@0.02 | 2.35% [0.00%, 6.69%] | 2.79% [0.00%, 7.80%] | 32.9% [31.1%, 36.1%] | 26.8% [24.8%, 30.3%] |
| pooled@0.05 | 4.97% [1.35%, 10.03%] | 5.31% [1.69%, 10.83%] | 75.1% [74.4%, 75.9%] | 70.9% [70.3%, 71.6%] |
| wg@0.005 | 0.84% [0.00%, 2.52%] | 1.12% [0.00%, 3.36%] | 13.1% [12.7%, 14.0%] | 12.0% [11.7%, 12.4%] |
| wg@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 18.3% [17.0%, 20.9%] | 15.7% [14.4%, 18.2%] |
| wg@0.02 | 2.01% [0.00%, 5.85%] | 2.23% [0.00%, 6.41%] | 27.4% [26.0%, 30.1%] | 21.1% [19.6%, 23.8%] |
| wg@0.05 | 4.86% [1.28%, 9.92%] | 5.03% [1.68%, 10.56%] | 74.5% [73.9%, 75.3%] | 70.3% [69.8%, 70.9%] |
| M1 | 3.69% [0.67%, 8.40%] | 4.19% [0.84%, 9.34%] | 63.5% [63.5%, 63.5%] | 60.5% [60.5%, 60.5%] |

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
| 0.5% | 13.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 18.3% | 0.24% [0.00%, 0.57%] | 0.24% [0.00%, 0.57%] | 0.00 |
| 2% | 27.4% | 0.00% [0.00%, 0.00%] | 0.04% [0.00%, 0.09%] | 0.00 |
| 5% | 74.5% | -0.22% [-0.71%, 0.12%] | -0.06% [-0.28%, 0.08%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.73% [0.22%, 1.34%] | 1.40% [0.56%, 2.53%] | 46.1% [45.9%, 46.5%] | 7.9% [7.6%, 8.3%] |
| pooled@0.01 | 1.68% [0.89%, 2.57%] | 3.63% [1.93%, 5.87%] | 53.2% [53.1%, 53.3%] | 18.2% [18.0%, 18.4%] |
| pooled@0.02 | 3.24% [1.84%, 4.85%] | 5.03% [3.06%, 7.56%] | 56.4% [56.3%, 56.4%] | 23.9% [23.6%, 24.3%] |
| pooled@0.05 | 7.65% [4.97%, 10.90%] | 13.97% [10.19%, 18.13%] | 63.4% [63.2%, 63.7%] | 37.3% [36.8%, 37.9%] |
| wg@0.005 | 0.22% [0.06%, 0.45%] | 0.56% [0.28%, 1.40%] | 41.5% [41.4%, 41.8%] | 5.4% [5.4%, 5.6%] |
| wg@0.01 | 0.95% [0.28%, 1.72%] | 1.68% [0.56%, 3.09%] | 48.2% [47.6%, 48.8%] | 11.0% [10.2%, 11.9%] |
| wg@0.02 | 2.35% [1.18%, 3.71%] | 3.63% [1.95%, 5.88%] | 55.5% [55.4%, 55.5%] | 20.8% [20.7%, 20.9%] |
| wg@0.05 | 6.20% [3.87%, 9.08%] | 10.34% [6.82%, 14.25%] | 61.1% [60.7%, 61.6%] | 33.7% [33.4%, 34.1%] |
| M1 | 6.59% [4.33%, 9.44%] | 12.01% [8.54%, 16.01%] | 62.7% [62.7%, 62.7%] | 34.2% [34.2%, 34.2%] |

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
| 0.5% | 41.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 48.2% | -0.15% [-0.72%, 0.46%] | 0.03% [-0.14%, 0.22%] | 0.00 |
| 2% | 55.5% | 0.00% [-0.71%, 0.04%] | -0.17% [-0.39%, 0.02%] | 0.00 |
| 5% | 61.1% | -0.44% [-1.63%, 0.57%] | 0.13% [-0.14%, 0.38%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.78%, 2.49%] | 3.91% [1.94%, 7.28%] | 15.4% [15.1%, 15.8%] | 11.4% [11.2%, 11.8%] |
| pooled@0.01 | 2.63% [1.50%, 4.01%] | 6.15% [3.37%, 10.73%] | 23.1% [22.6%, 23.7%] | 19.6% [19.0%, 20.4%] |
| pooled@0.02 | 5.25% [3.33%, 7.37%] | 10.89% [7.24%, 16.57%] | 33.6% [33.1%, 34.2%] | 34.1% [33.4%, 34.8%] |
| pooled@0.05 | 11.45% [8.20%, 15.06%] | 21.51% [16.80%, 28.29%] | 49.0% [48.6%, 49.5%] | 52.2% [51.8%, 52.7%] |
| wg@0.005 | 0.61% [0.17%, 1.12%] | 1.40% [0.56%, 3.62%] | 8.9% [8.8%, 9.0%] | 5.1% [5.0%, 5.3%] |
| wg@0.01 | 1.79% [0.89%, 2.82%] | 4.19% [1.96%, 7.54%] | 17.0% [16.9%, 17.0%] | 12.8% [12.7%, 12.9%] |
| wg@0.02 | 2.91% [1.58%, 4.48%] | 6.70% [3.63%, 11.76%] | 25.3% [24.6%, 26.2%] | 22.3% [21.3%, 23.6%] |
| wg@0.05 | 9.22% [6.37%, 12.45%] | 16.76% [13.26%, 23.60%] | 43.8% [43.3%, 44.6%] | 47.3% [46.8%, 48.0%] |
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
| 0.5% | 8.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 17.0% | 0.03% [-0.02%, 0.13%] | 0.03% [-0.01%, 0.10%] | 0.00 |
| 2% | 25.3% | -0.51% [-1.36%, 0.26%] | -0.23% [-0.51%, 0.02%] | 0.00 |
| 5% | 43.8% | -0.02% [-1.00%, 0.90%] | 0.00% [-0.28%, 0.27%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.95% [0.17%, 1.90%] | 1.68% [0.56%, 3.90%] | 80.9% [80.7%, 81.1%] | 16.3% [16.3%, 16.4%] |
| pooled@0.01 | 1.51% [0.45%, 2.82%] | 2.51% [1.11%, 4.78%] | 83.3% [83.1%, 83.6%] | 17.8% [17.5%, 18.0%] |
| pooled@0.02 | 2.46% [1.12%, 4.07%] | 3.35% [1.95%, 6.16%] | 89.0% [88.8%, 89.2%] | 26.1% [25.8%, 26.4%] |
| pooled@0.05 | 7.09% [4.82%, 9.50%] | 10.34% [7.87%, 14.09%] | 95.6% [95.5%, 95.7%] | 42.3% [42.1%, 42.6%] |
| wg@0.005 | 0.17% [0.00%, 0.45%] | 0.28% [0.00%, 1.12%] | 63.9% [63.5%, 64.6%] | 7.8% [7.6%, 8.2%] |
| wg@0.01 | 1.28% [0.39%, 2.46%] | 1.96% [0.84%, 4.16%] | 82.7% [82.6%, 82.8%] | 17.0% [17.0%, 17.1%] |
| wg@0.02 | 1.90% [0.68%, 3.41%] | 2.79% [1.13%, 5.07%] | 86.6% [86.2%, 87.0%] | 21.4% [20.8%, 22.0%] |
| wg@0.05 | 6.26% [4.26%, 8.30%] | 8.94% [6.92%, 12.29%] | 94.8% [94.7%, 95.0%] | 39.7% [39.3%, 40.1%] |
| M1 | 5.87% [4.20%, 7.65%] | 8.66% [6.80%, 12.32%] | 95.0% [95.0%, 95.0%] | 39.9% [39.9%, 39.9%] |

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
| 0.5% | 63.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 82.7% | -0.22% [-0.59%, 0.04%] | -0.03% [-0.10%, 0.04%] | 0.00 |
| 2% | 86.6% | 0.00% [-0.73%, 0.13%] | 0.02% [-0.08%, 0.12%] | 0.00 |
| 5% | 94.8% | -0.19% [-1.14%, 0.51%] | 0.12% [-0.09%, 0.36%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.73% [0.73%, 2.96%] | 5.03% [1.94%, 9.09%] | 26.3% [26.1%, 26.5%] | 16.9% [16.7%, 17.1%] |
| pooled@0.01 | 3.41% [1.90%, 5.20%] | 6.70% [4.18%, 10.31%] | 33.5% [33.3%, 33.7%] | 27.0% [26.7%, 27.5%] |
| pooled@0.02 | 7.04% [4.51%, 10.08%] | 14.80% [10.61%, 20.22%] | 43.5% [43.0%, 44.1%] | 42.1% [41.5%, 42.7%] |
| pooled@0.05 | 14.58% [10.75%, 18.61%] | 37.99% [30.47%, 45.51%] | 57.5% [57.0%, 58.0%] | 64.1% [63.3%, 65.0%] |
| wg@0.005 | 0.84% [0.23%, 1.61%] | 3.63% [1.12%, 6.93%] | 18.4% [18.3%, 18.5%] | 11.3% [11.2%, 11.4%] |
| wg@0.01 | 1.34% [0.45%, 2.41%] | 5.03% [1.68%, 9.09%] | 24.1% [23.8%, 24.4%] | 15.4% [15.2%, 15.6%] |
| wg@0.02 | 2.46% [1.06%, 4.16%] | 7.54% [2.82%, 13.13%] | 30.4% [30.1%, 30.8%] | 22.5% [22.0%, 23.3%] |
| wg@0.05 | 7.77% [5.03%, 11.08%] | 15.08% [10.96%, 22.63%] | 47.5% [46.8%, 48.3%] | 47.2% [46.1%, 48.5%] |
| M1 | 10.56% [7.78%, 13.56%] | 29.61% [23.31%, 35.96%] | 51.1% [51.1%, 51.1%] | 53.4% [53.4%, 53.4%] |

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
| 1% | 24.1% | 0.11% [0.00%, 0.35%] | -0.05% [-0.15%, 0.05%] | 0.00 |
| 2% | 30.4% | 1.94% [-1.55%, 4.54%] | -0.26% [-0.73%, 0.19%] | 0.00 |
| 5% | 47.5% | -6.78% [-8.78%, -0.61%] | -1.32% [-2.02%, -0.78%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)
