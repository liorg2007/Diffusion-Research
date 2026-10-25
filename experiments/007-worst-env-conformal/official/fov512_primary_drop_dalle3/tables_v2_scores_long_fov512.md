# H2-v2 tables (scores_long_fov512)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 13.7% [12.4%, 16.3%] | 10.2% [9.4%, 12.0%] |
| pooled@0.01 | 1.68% [0.00%, 5.04%] | 1.68% [0.00%, 5.04%] | 18.7% [17.5%, 21.1%] | 13.5% [12.6%, 15.3%] |
| pooled@0.02 | 2.29% [0.00%, 6.52%] | 2.51% [0.00%, 7.26%] | 40.2% [39.2%, 42.1%] | 28.1% [26.9%, 30.4%] |
| pooled@0.05 | 5.25% [1.35%, 10.76%] | 5.59% [1.69%, 11.67%] | 74.7% [73.3%, 76.2%] | 70.7% [69.6%, 71.9%] |
| wg@0.005 | 0.89% [0.00%, 2.69%] | 1.12% [0.00%, 3.36%] | 11.7% [11.0%, 13.1%] | 8.8% [8.4%, 9.6%] |
| wg@0.01 | 1.56% [0.00%, 4.71%] | 1.68% [0.00%, 5.04%] | 15.1% [13.8%, 17.7%] | 11.1% [10.2%, 12.9%] |
| wg@0.02 | 2.23% [0.00%, 6.35%] | 2.51% [0.00%, 7.26%] | 37.2% [36.3%, 38.9%] | 25.5% [24.5%, 27.6%] |
| wg@0.05 | 5.03% [1.24%, 10.45%] | 5.31% [1.68%, 11.26%] | 72.4% [70.9%, 74.0%] | 68.1% [66.8%, 69.6%] |
| M1 | 3.85% [0.73%, 8.74%] | 4.19% [0.84%, 9.39%] | 63.2% [63.2%, 63.2%] | 57.0% [57.0%, 57.0%] |

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
| 0.5% | 11.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 15.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 37.2% | 0.01% [0.00%, 0.03%] | 0.01% [0.00%, 0.02%] | 0.00 |
| 5% | 72.4% | 0.04% [-0.26%, 0.59%] | 0.08% [-0.08%, 0.22%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.51% [0.00%, 4.54%] | 1.68% [0.00%, 5.04%] | 13.2% [12.2%, 15.2%] | 10.1% [9.4%, 11.6%] |
| pooled@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 19.1% [18.0%, 21.5%] | 13.4% [12.2%, 15.7%] |
| pooled@0.02 | 2.35% [0.00%, 6.69%] | 2.79% [0.00%, 7.80%] | 30.3% [28.4%, 33.6%] | 22.2% [20.1%, 25.8%] |
| pooled@0.05 | 4.97% [1.35%, 10.03%] | 5.31% [1.69%, 10.83%] | 74.3% [73.7%, 75.1%] | 69.2% [68.6%, 69.9%] |
| wg@0.005 | 0.84% [0.00%, 2.52%] | 1.12% [0.00%, 3.36%] | 11.0% [10.6%, 11.8%] | 8.1% [7.9%, 8.6%] |
| wg@0.01 | 1.79% [0.00%, 5.38%] | 1.96% [0.00%, 5.88%] | 15.4% [14.1%, 18.1%] | 11.8% [10.6%, 14.2%] |
| wg@0.02 | 2.01% [0.00%, 5.85%] | 2.23% [0.00%, 6.41%] | 24.8% [23.4%, 27.5%] | 16.6% [15.0%, 19.4%] |
| wg@0.05 | 4.86% [1.28%, 9.92%] | 5.03% [1.68%, 10.56%] | 73.9% [73.4%, 74.6%] | 68.6% [68.1%, 69.2%] |
| M1 | 3.69% [0.67%, 8.40%] | 4.19% [0.84%, 9.34%] | 62.8% [62.8%, 62.8%] | 57.2% [57.2%, 57.2%] |

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
| 0.5% | 11.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 15.4% | 0.24% [0.00%, 0.57%] | 0.24% [0.00%, 0.57%] | 0.00 |
| 2% | 24.8% | 0.00% [0.00%, 0.00%] | 0.03% [0.00%, 0.07%] | 0.00 |
| 5% | 73.9% | -0.23% [-0.73%, 0.10%] | -0.07% [-0.29%, 0.07%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.73% [0.22%, 1.34%] | 1.40% [0.56%, 2.53%] | 49.1% [48.7%, 49.5%] | 7.7% [7.4%, 8.1%] |
| pooled@0.01 | 1.68% [0.89%, 2.57%] | 3.63% [1.93%, 5.87%] | 57.2% [57.1%, 57.4%] | 16.3% [16.2%, 16.5%] |
| pooled@0.02 | 3.24% [1.84%, 4.85%] | 5.03% [3.06%, 7.56%] | 60.4% [60.4%, 60.5%] | 21.5% [21.2%, 21.9%] |
| pooled@0.05 | 7.65% [4.97%, 10.90%] | 13.97% [10.19%, 18.13%] | 66.3% [66.1%, 66.5%] | 35.2% [34.6%, 35.9%] |
| wg@0.005 | 0.22% [0.06%, 0.45%] | 0.56% [0.28%, 1.40%] | 43.7% [43.6%, 44.0%] | 5.4% [5.3%, 5.5%] |
| wg@0.01 | 0.95% [0.28%, 1.72%] | 1.68% [0.56%, 3.09%] | 51.6% [50.9%, 52.2%] | 10.1% [9.4%, 10.9%] |
| wg@0.02 | 2.35% [1.18%, 3.71%] | 3.63% [1.95%, 5.88%] | 59.8% [59.7%, 59.8%] | 18.9% [18.8%, 19.0%] |
| wg@0.05 | 6.20% [3.87%, 9.08%] | 10.34% [6.82%, 14.25%] | 64.1% [63.8%, 64.4%] | 31.5% [31.0%, 31.9%] |
| M1 | 6.59% [4.33%, 9.44%] | 12.01% [8.54%, 16.01%] | 65.1% [65.1%, 65.1%] | 32.1% [32.1%, 32.1%] |

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
| 0.5% | 43.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 51.6% | -0.13% [-0.69%, 0.49%] | 0.04% [-0.13%, 0.24%] | 0.00 |
| 2% | 59.8% | 0.00% [-0.72%, 0.03%] | -0.18% [-0.41%, 0.02%] | 0.00 |
| 5% | 64.1% | -0.47% [-1.57%, 0.57%] | 0.12% [-0.13%, 0.39%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.56% [0.78%, 2.49%] | 3.91% [1.94%, 7.28%] | 13.2% [13.0%, 13.6%] | 8.3% [8.1%, 8.7%] |
| pooled@0.01 | 2.63% [1.50%, 4.01%] | 6.15% [3.37%, 10.73%] | 20.5% [20.1%, 21.1%] | 15.5% [14.9%, 16.3%] |
| pooled@0.02 | 5.25% [3.33%, 7.37%] | 10.89% [7.24%, 16.57%] | 30.4% [29.9%, 31.0%] | 28.8% [28.2%, 29.5%] |
| pooled@0.05 | 11.45% [8.20%, 15.06%] | 21.51% [16.80%, 28.29%] | 45.6% [45.1%, 46.1%] | 47.3% [46.9%, 47.9%] |
| wg@0.005 | 0.61% [0.17%, 1.12%] | 1.40% [0.56%, 3.62%] | 7.5% [7.4%, 7.6%] | 3.5% [3.4%, 3.6%] |
| wg@0.01 | 1.79% [0.89%, 2.82%] | 4.19% [1.96%, 7.54%] | 14.7% [14.6%, 14.8%] | 9.6% [9.5%, 9.7%] |
| wg@0.02 | 2.91% [1.58%, 4.48%] | 6.70% [3.63%, 11.76%] | 22.6% [22.0%, 23.3%] | 17.8% [16.8%, 19.1%] |
| wg@0.05 | 9.22% [6.37%, 12.45%] | 16.76% [13.26%, 23.60%] | 40.6% [40.1%, 41.3%] | 41.9% [41.3%, 42.7%] |
| M1 | 0.06% [0.00%, 0.17%] | 0.28% [0.00%, 0.84%] | 0.8% [0.8%, 0.8%] | 0.7% [0.7%, 0.7%] |

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
| 1% | 14.7% | 0.02% [-0.05%, 0.11%] | 0.02% [-0.02%, 0.09%] | 0.00 |
| 2% | 22.6% | -0.60% [-1.52%, 0.22%] | -0.27% [-0.57%, -0.01%] | 0.00 |
| 5% | 40.6% | -0.10% [-1.09%, 0.81%] | -0.04% [-0.32%, 0.23%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.95% [0.17%, 1.90%] | 1.68% [0.56%, 3.90%] | 80.4% [80.3%, 80.7%] | 16.3% [16.3%, 16.4%] |
| pooled@0.01 | 1.51% [0.45%, 2.82%] | 2.51% [1.11%, 4.78%] | 83.3% [83.1%, 83.6%] | 17.6% [17.4%, 17.8%] |
| pooled@0.02 | 2.46% [1.12%, 4.07%] | 3.35% [1.95%, 6.16%] | 89.7% [89.4%, 89.9%] | 24.5% [24.3%, 24.8%] |
| pooled@0.05 | 7.09% [4.82%, 9.50%] | 10.34% [7.87%, 14.09%] | 95.6% [95.6%, 95.7%] | 40.2% [39.9%, 40.5%] |
| wg@0.005 | 0.17% [0.00%, 0.45%] | 0.28% [0.00%, 1.12%] | 64.0% [63.6%, 64.7%] | 8.0% [7.8%, 8.4%] |
| wg@0.01 | 1.28% [0.39%, 2.46%] | 1.96% [0.84%, 4.16%] | 82.7% [82.6%, 82.8%] | 17.0% [17.0%, 17.1%] |
| wg@0.02 | 1.90% [0.68%, 3.41%] | 2.79% [1.13%, 5.07%] | 86.7% [86.3%, 87.2%] | 20.5% [20.0%, 21.0%] |
| wg@0.05 | 6.26% [4.26%, 8.30%] | 8.94% [6.92%, 12.29%] | 94.9% [94.8%, 95.1%] | 37.7% [37.3%, 38.1%] |
| M1 | 5.87% [4.20%, 7.65%] | 8.66% [6.80%, 12.32%] | 95.0% [95.0%, 95.0%] | 37.8% [37.8%, 37.8%] |

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
| 0.5% | 64.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 82.7% | -0.23% [-0.61%, 0.04%] | -0.03% [-0.11%, 0.03%] | 0.00 |
| 2% | 86.7% | 0.00% [-0.71%, 0.13%] | 0.03% [-0.08%, 0.12%] | 0.00 |
| 5% | 94.9% | -0.45% [-1.45%, 0.29%] | -0.04% [-0.27%, 0.22%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 1.73% [0.73%, 2.96%] | 5.03% [1.94%, 9.09%] | 30.1% [29.9%, 30.4%] | 16.4% [16.2%, 16.5%] |
| pooled@0.01 | 3.41% [1.90%, 5.20%] | 6.70% [4.18%, 10.31%] | 39.1% [38.9%, 39.5%] | 26.4% [26.0%, 26.9%] |
| pooled@0.02 | 7.04% [4.51%, 10.08%] | 14.80% [10.61%, 20.22%] | 50.4% [49.8%, 51.1%] | 42.1% [41.6%, 42.7%] |
| pooled@0.05 | 14.58% [10.75%, 18.61%] | 37.99% [30.47%, 45.51%] | 65.8% [65.3%, 66.3%] | 63.7% [63.0%, 64.5%] |
| wg@0.005 | 0.84% [0.23%, 1.61%] | 3.63% [1.12%, 6.93%] | 21.0% [20.9%, 21.2%] | 10.9% [10.8%, 11.0%] |
| wg@0.01 | 1.34% [0.45%, 2.41%] | 5.03% [1.68%, 9.09%] | 27.6% [27.3%, 28.0%] | 14.8% [14.7%, 15.0%] |
| wg@0.02 | 2.46% [1.06%, 4.16%] | 7.54% [2.82%, 13.13%] | 35.2% [34.9%, 35.8%] | 22.0% [21.5%, 22.6%] |
| wg@0.05 | 7.77% [5.03%, 11.08%] | 15.08% [10.96%, 22.63%] | 54.5% [53.6%, 55.5%] | 47.3% [46.2%, 48.6%] |
| M1 | 10.56% [7.78%, 13.56%] | 29.61% [23.31%, 35.96%] | 59.5% [59.5%, 59.5%] | 53.4% [53.4%, 53.4%] |

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
| 0.5% | 21.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 27.6% | 0.10% [0.00%, 0.33%] | -0.05% [-0.16%, 0.04%] | 0.00 |
| 2% | 35.2% | 1.96% [-1.52%, 4.54%] | -0.24% [-0.71%, 0.21%] | 0.00 |
| 5% | 54.5% | -6.14% [-7.97%, -0.19%] | -1.13% [-1.82%, -0.60%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)
