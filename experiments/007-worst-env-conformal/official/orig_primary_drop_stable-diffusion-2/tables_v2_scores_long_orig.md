# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 27.7% [27.0%, 28.7%] | 20.4% [19.9%, 21.2%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 37.1% [36.4%, 38.3%] | 28.0% [27.3%, 29.3%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 48.6% [47.6%, 50.3%] | 39.7% [38.7%, 41.6%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 76.1% [75.8%, 76.6%] | 69.1% [68.7%, 69.7%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 15.9% [15.5%, 16.7%] | 9.0% [8.7%, 9.6%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 27.0% [26.3%, 28.1%] | 20.0% [19.5%, 21.0%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 39.1% [37.9%, 41.0%] | 30.1% [28.9%, 32.1%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 70.3% [69.6%, 71.3%] | 62.7% [61.7%, 63.9%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 43.4% [43.4%, 43.4%] | 35.7% [35.7%, 35.7%] |

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
| 0.5% | 15.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 27.0% | 0.08% [0.00%, 0.11%] | 0.09% [-0.09%, 0.23%] | 0.00 |
| 2% | 39.1% | 0.36% [0.00%, 1.00%] | 0.18% [0.00%, 0.42%] | 0.00 |
| 5% | 70.3% | -0.28% [-0.79%, 0.36%] | 0.01% [-0.17%, 0.18%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 22.0% [21.5%, 22.6%] | 13.4% [12.9%, 14.1%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 28.7% [27.3%, 31.3%] | 20.0% [18.5%, 22.7%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 50.2% [49.1%, 52.0%] | 41.9% [40.7%, 43.7%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 70.3% [70.0%, 70.8%] | 62.4% [62.0%, 62.9%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 15.4% [15.3%, 15.6%] | 6.9% [6.8%, 7.1%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 23.2% [22.4%, 24.8%] | 14.7% [13.9%, 16.3%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 35.2% [33.8%, 37.8%] | 27.5% [26.2%, 29.9%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 67.3% [67.0%, 67.7%] | 59.6% [59.3%, 60.1%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 56.2% [56.2%, 56.2%] | 47.5% [47.5%, 47.5%] |

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
| 0.5% | 15.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 23.2% | 0.31% [0.00%, 0.51%] | 0.18% [-0.17%, 0.49%] | 0.00 |
| 2% | 35.2% | 0.00% [-0.18%, 0.00%] | -0.06% [-0.20%, 0.08%] | 0.00 |
| 5% | 67.3% | 0.10% [-0.84%, 0.33%] | -0.24% [-0.51%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 60.6% [60.3%, 61.0%] | 13.3% [13.1%, 13.5%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 67.3% [67.0%, 67.6%] | 17.9% [17.7%, 18.1%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 77.6% [77.4%, 78.0%] | 27.7% [27.1%, 28.3%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 86.8% [86.7%, 87.0%] | 45.0% [44.8%, 45.2%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 33.0% [31.9%, 35.0%] | 5.3% [5.0%, 5.9%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 60.0% [59.8%, 60.3%] | 13.2% [13.1%, 13.4%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 65.5% [64.8%, 66.3%] | 17.8% [17.4%, 18.2%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 86.1% [85.9%, 86.3%] | 43.0% [42.6%, 43.5%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 85.5% [85.5%, 85.5%] | 40.1% [40.1%, 40.1%] |

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
| 0.5% | 33.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 60.0% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 65.5% | 0.18% [0.00%, 0.58%] | 0.02% [-0.06%, 0.11%] | 0.00 |
| 5% | 86.1% | -0.84% [-1.96%, 0.01%] | -0.22% [-0.50%, -0.00%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 9.9% [9.7%, 10.1%] | 14.1% [13.9%, 14.4%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 16.1% [15.9%, 16.2%] | 21.2% [21.1%, 21.4%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 22.6% [22.3%, 23.2%] | 28.3% [27.9%, 28.8%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 40.9% [40.7%, 41.1%] | 47.6% [47.4%, 47.9%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 3.7% [3.7%, 3.8%] | 6.8% [6.8%, 6.8%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 9.8% [9.6%, 10.3%] | 13.7% [13.4%, 14.3%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 16.6% [16.3%, 17.0%] | 21.8% [21.4%, 22.2%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 31.9% [31.5%, 32.4%] | 38.9% [38.5%, 39.6%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 28.8% [28.8%, 28.8%] | 35.4% [35.4%, 35.4%] |

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
| 0.5% | 3.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 9.8% | -2.87% [-4.91%, -1.12%] | -0.51% [-0.91%, -0.19%] | 0.00 |
| 2% | 16.6% | -2.13% [-3.29%, -1.00%] | -0.37% [-0.67%, -0.05%] | 0.00 |
| 5% | 31.9% | -1.88% [-3.24%, -0.68%] | -0.33% [-0.72%, 0.05%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 76.9% [76.2%, 78.2%] | 20.1% [19.4%, 21.4%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 87.9% [87.6%, 88.3%] | 31.0% [30.6%, 31.7%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 93.5% [93.4%, 93.7%] | 40.0% [39.5%, 40.6%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 97.9% [97.8%, 98.0%] | 54.6% [54.4%, 54.9%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 68.7% [68.3%, 69.5%] | 13.9% [13.6%, 14.5%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 74.6% [73.8%, 76.0%] | 18.1% [17.3%, 19.6%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 85.7% [85.1%, 86.6%] | 29.7% [29.0%, 30.8%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 96.4% [96.3%, 96.5%] | 48.3% [47.9%, 48.8%] |
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
| 0.5% | 68.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 74.6% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 85.7% | 2.33% [0.67%, 3.82%] | 0.52% [0.11%, 0.95%] | 0.00 |
| 5% | 96.4% | -1.40% [-2.54%, 0.17%] | -0.36% [-0.70%, -0.08%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 54.9% [54.7%, 55.1%] | 29.9% [29.5%, 30.4%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 57.9% [57.6%, 58.3%] | 36.8% [36.2%, 37.5%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 65.6% [65.5%, 65.7%] | 51.2% [51.0%, 51.4%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 70.1% [70.1%, 70.2%] | 64.2% [64.0%, 64.3%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 50.4% [50.3%, 50.7%] | 19.6% [19.2%, 20.2%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 55.6% [55.5%, 55.8%] | 30.2% [29.8%, 30.8%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 58.8% [58.5%, 59.3%] | 38.4% [37.8%, 39.4%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 67.3% [67.3%, 67.4%] | 56.1% [55.9%, 56.3%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 56.9% [56.9%, 56.9%] | 35.5% [35.5%, 35.5%] |

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
| 0.5% | 50.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 55.6% | 0.14% [0.00%, 0.43%] | 0.00% [-0.08%, 0.08%] | 0.00 |
| 2% | 58.8% | 1.94% [0.64%, 3.59%] | 0.32% [0.07%, 0.65%] | 0.00 |
| 5% | 67.3% | 0.27% [-0.38%, 1.11%] | -0.06% [-0.27%, 0.15%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
