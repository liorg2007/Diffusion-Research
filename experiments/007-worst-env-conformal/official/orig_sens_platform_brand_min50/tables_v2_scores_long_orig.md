# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform_brand, platform.

## B0 / grouping platform_brand  [single model 'full' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 5.22% [0.00%, 15.07%] | 29.0% [28.4%, 30.0%] | 21.3% [20.8%, 22.2%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 6.09% [0.86%, 18.05%] | 38.6% [38.0%, 39.8%] | 29.1% [28.4%, 30.3%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 7.83% [2.33%, 23.76%] | 50.2% [49.2%, 51.9%] | 40.7% [39.7%, 42.6%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 10.43% [7.02%, 25.59%] | 77.0% [76.7%, 77.5%] | 69.4% [69.0%, 70.0%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 2.61% [0.00%, 6.82%] | 16.9% [16.5%, 17.7%] | 9.2% [8.9%, 9.9%] |
| wg@0.01 | 0.34% [0.00%, 0.89%] | 2.61% [0.00%, 6.82%] | 16.9% [16.5%, 17.7%] | 9.2% [8.9%, 9.9%] |
| wg@0.02 | 0.34% [0.00%, 0.89%] | 2.61% [0.00%, 6.82%] | 25.5% [25.3%, 25.7%] | 16.6% [16.3%, 17.0%] |
| wg@0.05 | 1.79% [0.11%, 4.88%] | 6.96% [1.37%, 20.83%] | 42.4% [41.5%, 44.0%] | 32.8% [32.0%, 34.5%] |
| M1 | 1.34% [0.11%, 3.38%] | 5.22% [1.39%, 15.07%] | 44.8% [44.8%, 44.8%] | 36.8% [36.8%, 36.8%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 1.0, "0.01": 0.57, "0.02": 0.04, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 0.84% | 2.23% | 1.68% | 1.40% | 0.56% |
| pooled@0.01 | 0.28% | 2.51% | 1.12% | 1.40% | 1.12% |
| wg@0.01 | 0.00% | 1.12% | 0.00% | 0.56% | 0.00% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 16.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 16.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 25.5% | -1.29% [-4.22%, 0.00%] | -0.08% [-0.24%, 0.00%] | 0.00 |
| 5% | 42.4% | -0.23% [-0.81%, 1.47%] | -0.02% [-0.18%, 0.14%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/FBH/Samsung, M1/group/FBH/other, M1/group/WA/other, M1/group/jpeg75_resize05/other, M1/group/native/other, pooled@0.01/platform/WA, pooled@0.01/group/FBH/other, pooled@0.01/group/WA/other, pooled@0.01/group/jpeg75_resize05/other, pooled@0.01/group/screenshot_sim/other)

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 29.0% [28.4%, 30.0%] | 21.3% [20.8%, 22.2%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 38.6% [38.0%, 39.8%] | 29.1% [28.4%, 30.3%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 50.2% [49.2%, 51.9%] | 40.7% [39.7%, 42.6%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 77.0% [76.7%, 77.5%] | 69.4% [69.0%, 70.0%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 16.9% [16.5%, 17.7%] | 9.2% [8.9%, 9.9%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 28.2% [27.5%, 29.3%] | 20.8% [20.3%, 21.8%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 40.5% [39.4%, 42.4%] | 31.2% [30.0%, 33.2%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 71.6% [70.8%, 72.5%] | 63.5% [62.6%, 64.6%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 44.8% [44.8%, 44.8%] | 36.8% [36.8%, 36.8%] |

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
| 0.5% | 16.9% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 28.2% | 0.10% [0.00%, 0.15%] | 0.12% [-0.08%, 0.28%] | 0.00 |
| 2% | 40.5% | 0.35% [0.00%, 0.98%] | 0.17% [0.00%, 0.38%] | 0.00 |
| 5% | 71.6% | -0.30% [-0.81%, 0.34%] | -0.00% [-0.18%, 0.17%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform_brand  [single model 'full' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 3.48% [0.00%, 9.59%] | 24.7% [24.2%, 25.3%] | 14.9% [14.3%, 15.8%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 6.96% [0.00%, 20.83%] | 31.4% [30.0%, 34.0%] | 22.6% [21.2%, 25.2%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 7.83% [1.75%, 23.76%] | 52.5% [51.4%, 54.2%] | 43.6% [42.5%, 45.4%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 11.30% [7.02%, 26.22%] | 72.4% [72.1%, 72.9%] | 64.3% [63.9%, 64.8%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 1.15% [0.00%, 3.45%] | 17.3% [17.1%, 17.5%] | 7.8% [7.6%, 8.0%] |
| wg@0.01 | 0.22% [0.00%, 0.56%] | 1.15% [0.00%, 3.45%] | 17.3% [17.1%, 17.5%] | 7.8% [7.6%, 8.0%] |
| wg@0.02 | 0.28% [0.00%, 0.72%] | 1.15% [0.00%, 3.45%] | 18.9% [18.6%, 19.4%] | 8.9% [8.6%, 9.5%] |
| wg@0.05 | 2.07% [0.11%, 5.65%] | 7.83% [0.88%, 23.76%] | 39.7% [38.4%, 42.3%] | 32.0% [30.8%, 34.5%] |
| M1 | 2.29% [0.56%, 5.17%] | 6.96% [2.35%, 20.83%] | 58.6% [58.6%, 58.6%] | 48.6% [48.6%, 48.6%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 1.0, "0.01": 0.57, "0.02": 0.04, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.40% | 2.79% | 2.23% | 3.07% | 1.96% |
| pooled@0.01 | 0.56% | 2.51% | 1.68% | 1.68% | 1.40% |
| wg@0.01 | 0.00% | 0.56% | 0.00% | 0.56% | 0.00% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 17.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 17.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 18.9% | -0.57% [-1.42%, 0.00%] | -0.01% [-0.09%, 0.09%] | 0.00 |
| 5% | 39.7% | 0.87% [0.00%, 2.94%] | 0.06% [-0.17%, 0.37%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH/Samsung, M1/group/FBH/other, M1/group/WA/other, M1/group/jpeg75_resize05/Apple, M1/group/jpeg75_resize05/Samsung, M1/group/jpeg75_resize05/other, M1/group/native/Samsung, M1/group/native/other, M1/group/screenshot_sim/Samsung, M1/group/screenshot_sim/other, pooled@0.01/platform/WA, pooled@0.01/group/FBH/other, pooled@0.01/group/WA/other, pooled@0.01/group/jpeg75_resize05/other, pooled@0.01/group/screenshot_sim/other)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 24.7% [24.2%, 25.3%] | 14.9% [14.3%, 15.8%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 31.4% [30.0%, 34.0%] | 22.6% [21.2%, 25.2%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 52.5% [51.4%, 54.2%] | 43.6% [42.5%, 45.4%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 72.4% [72.1%, 72.9%] | 64.3% [63.9%, 64.8%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 17.3% [17.1%, 17.5%] | 7.8% [7.6%, 8.0%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 26.0% [25.2%, 27.5%] | 16.1% [15.3%, 17.8%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 37.5% [36.1%, 40.0%] | 29.9% [28.6%, 32.2%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 69.4% [69.1%, 69.9%] | 61.2% [60.9%, 61.7%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 58.6% [58.6%, 58.6%] | 48.6% [48.6%, 48.6%] |

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
| 0.5% | 17.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 26.0% | 0.30% [0.00%, 0.48%] | 0.17% [-0.17%, 0.47%] | 0.00 |
| 2% | 37.5% | 0.00% [-0.17%, 0.00%] | -0.06% [-0.20%, 0.08%] | 0.00 |
| 5% | 69.4% | 0.11% [-0.80%, 0.34%] | -0.23% [-0.49%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 8.77% [2.38%, 14.29%] | 62.8% [62.5%, 63.2%] | 12.8% [12.6%, 13.0%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 10.53% [3.51%, 14.29%] | 69.4% [69.1%, 69.7%] | 17.1% [16.9%, 17.3%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 12.28% [4.76%, 21.43%] | 79.6% [79.3%, 79.9%] | 26.4% [25.9%, 27.0%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 15.79% [8.53%, 42.86%] | 88.1% [87.9%, 88.3%] | 43.6% [43.4%, 43.8%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 7.02% [0.00%, 14.29%] | 34.1% [33.0%, 36.2%] | 5.1% [4.8%, 5.6%] |
| wg@0.01 | 0.34% [0.00%, 0.68%] | 7.02% [0.00%, 14.29%] | 34.1% [33.0%, 36.2%] | 5.1% [4.8%, 5.6%] |
| wg@0.02 | 0.34% [0.00%, 0.68%] | 7.02% [0.00%, 14.29%] | 34.1% [33.0%, 36.2%] | 5.1% [4.8%, 5.6%] |
| wg@0.05 | 0.56% [0.11%, 1.06%] | 10.53% [2.82%, 14.29%] | 59.7% [59.2%, 60.3%] | 11.5% [11.2%, 11.8%] |
| M1 | 3.80% [2.23%, 5.53%] | 14.04% [7.14%, 35.71%] | 86.9% [86.9%, 86.9%] | 38.6% [38.6%, 38.6%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 1.0, "0.01": 0.57, "0.02": 0.04, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 3.35% | 4.47% | 2.79% | 4.19% | 4.19% |
| pooled@0.01 | 0.00% | 3.07% | 0.28% | 0.84% | 0.56% |
| wg@0.01 | 0.00% | 1.68% | 0.00% | 0.00% | 0.00% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 34.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 34.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 34.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 5% | 59.7% | 3.51% [0.00%, 10.71%] | 0.17% [-0.11%, 0.45%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH/Apple, M1/group/FBH/Huawei, M1/group/WA/Apple, M1/group/WA/Huawei, M1/group/WA/other, M1/group/jpeg75_resize05/Huawei, M1/group/jpeg75_resize05/Samsung, M1/group/jpeg75_resize05/other, M1/group/native/Huawei, M1/group/screenshot_sim/Apple, M1/group/screenshot_sim/Huawei, M1/group/screenshot_sim/other, pooled@0.01/platform/WA, pooled@0.01/group/WA/Apple, pooled@0.01/group/WA/Huawei, pooled@0.01/group/screenshot_sim/Huawei)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 62.8% [62.5%, 63.2%] | 12.8% [12.6%, 13.0%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 69.4% [69.1%, 69.7%] | 17.1% [16.9%, 17.3%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 79.6% [79.3%, 79.9%] | 26.4% [25.9%, 27.0%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 88.1% [87.9%, 88.3%] | 43.6% [43.4%, 43.8%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 34.1% [33.0%, 36.2%] | 5.1% [4.8%, 5.6%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 62.3% [62.1%, 62.5%] | 12.7% [12.6%, 12.9%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 67.5% [66.7%, 68.3%] | 17.0% [16.7%, 17.4%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 87.4% [87.3%, 87.6%] | 41.8% [41.4%, 42.2%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 86.9% [86.9%, 86.9%] | 38.6% [38.6%, 38.6%] |

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
| 0.5% | 34.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 62.3% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 67.5% | 0.21% [0.00%, 0.67%] | 0.03% [-0.04%, 0.13%] | 0.00 |
| 5% | 87.4% | -0.84% [-1.96%, 0.01%] | -0.23% [-0.50%, -0.00%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 26.32% [15.25%, 57.14%] | 11.3% [11.1%, 11.5%] | 14.7% [14.5%, 15.0%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 40.35% [27.72%, 71.43%] | 17.1% [17.0%, 17.3%] | 21.8% [21.7%, 22.0%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 40.35% [31.40%, 71.43%] | 23.0% [22.7%, 23.5%] | 28.8% [28.4%, 29.4%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 56.14% [52.33%, 72.41%] | 41.0% [40.8%, 41.2%] | 49.0% [48.7%, 49.2%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 6.09% [3.79%, 11.30%] | 5.6% [5.6%, 5.7%] | 6.9% [6.9%, 6.9%] |
| wg@0.01 | 1.06% [0.50%, 1.79%] | 6.09% [3.79%, 11.30%] | 5.6% [5.6%, 5.7%] | 6.9% [6.9%, 6.9%] |
| wg@0.02 | 1.51% [0.56%, 2.87%] | 10.43% [4.65%, 21.43%] | 6.4% [6.1%, 6.8%] | 7.8% [7.4%, 8.4%] |
| wg@0.05 | 6.59% [4.57%, 8.86%] | 36.84% [25.86%, 64.29%] | 16.1% [15.9%, 16.2%] | 20.4% [20.2%, 20.6%] |
| M1 | 11.12% [8.63%, 13.87%] | 49.12% [38.89%, 71.43%] | 28.5% [28.5%, 28.5%] | 35.7% [35.7%, 35.8%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 1.0, "0.01": 0.57, "0.02": 0.04, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 4.19% | 1.40% | 1.96% | 38.27% | 9.78% |
| pooled@0.01 | 2.51% | 0.28% | 0.28% | 27.37% | 6.70% |
| wg@0.01 | 1.12% | 0.00% | 0.00% | 3.35% | 0.84% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 5.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 5.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 6.4% | -2.45% [-5.55%, -0.59%] | -0.37% [-0.60%, -0.19%] | 0.00 |
| 5% | 16.1% | -2.70% [-5.72%, 0.00%] | -0.58% [-0.90%, -0.30%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH/Huawei, M1/group/FBH/other, M1/group/WA/other, M1/group/jpeg75_resize05/Apple, M1/group/jpeg75_resize05/Huawei, M1/group/jpeg75_resize05/Samsung, M1/group/jpeg75_resize05/other, M1/group/native/Huawei, M1/group/native/Samsung, M1/group/native/other, M1/group/screenshot_sim/Apple, M1/group/screenshot_sim/Huawei, M1/group/screenshot_sim/Samsung, M1/group/screenshot_sim/other, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05/Apple, pooled@0.01/group/jpeg75_resize05/Huawei, pooled@0.01/group/jpeg75_resize05/Samsung, pooled@0.01/group/jpeg75_resize05/other, pooled@0.01/group/native/Huawei, pooled@0.01/group/native/other, pooled@0.01/group/screenshot_sim/Apple, pooled@0.01/group/screenshot_sim/Huawei, pooled@0.01/group/screenshot_sim/Samsung, pooled@0.01/group/screenshot_sim/other)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 11.3% [11.1%, 11.5%] | 14.7% [14.5%, 15.0%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 17.1% [17.0%, 17.3%] | 21.8% [21.7%, 22.0%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 23.0% [22.7%, 23.5%] | 28.8% [28.4%, 29.4%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 41.0% [40.8%, 41.2%] | 49.0% [48.7%, 49.2%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 5.6% [5.6%, 5.7%] | 6.9% [6.9%, 6.9%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 11.2% [11.0%, 11.7%] | 14.4% [14.1%, 15.0%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 17.7% [17.4%, 18.0%] | 22.3% [22.0%, 22.7%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 31.7% [31.3%, 32.3%] | 39.6% [39.2%, 40.3%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 28.5% [28.5%, 28.5%] | 35.7% [35.7%, 35.8%] |

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
| 0.5% | 5.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 11.2% | -2.94% [-4.96%, -1.13%] | -0.53% [-0.92%, -0.20%] | 0.00 |
| 2% | 17.7% | -2.18% [-3.36%, -1.06%] | -0.39% [-0.70%, -0.06%] | 0.00 |
| 5% | 31.7% | -1.93% [-3.34%, -0.70%] | -0.36% [-0.76%, 0.03%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 10.53% [0.00%, 35.71%] | 79.5% [78.9%, 80.6%] | 20.4% [19.6%, 21.8%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 12.28% [1.39%, 42.86%] | 89.4% [89.1%, 89.8%] | 32.6% [32.1%, 33.4%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 19.30% [4.85%, 50.00%] | 94.3% [94.2%, 94.5%] | 42.9% [42.4%, 43.5%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 24.56% [12.00%, 64.29%] | 98.1% [98.1%, 98.2%] | 58.2% [57.9%, 58.5%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 10.53% [0.00%, 35.71%] | 72.1% [71.7%, 72.8%] | 13.5% [13.1%, 14.2%] |
| wg@0.01 | 0.50% [0.00%, 1.42%] | 10.53% [0.00%, 35.71%] | 72.1% [71.7%, 72.8%] | 13.5% [13.1%, 14.2%] |
| wg@0.02 | 0.50% [0.00%, 1.42%] | 10.53% [0.00%, 35.71%] | 72.1% [71.7%, 72.9%] | 13.5% [13.1%, 14.2%] |
| wg@0.05 | 1.01% [0.00%, 2.84%] | 10.53% [0.00%, 35.71%] | 74.0% [73.2%, 75.5%] | 16.3% [15.3%, 18.2%] |
| M1 | 2.29% [0.84%, 4.28%] | 15.79% [4.23%, 35.71%] | 93.0% [93.0%, 93.0%] | 40.2% [40.2%, 40.2%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 1.0, "0.01": 0.57, "0.02": 0.04, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 2.23% | 1.12% | 1.12% | 3.07% | 3.91% |
| pooled@0.01 | 2.23% | 0.56% | 0.28% | 1.40% | 2.51% |
| wg@0.01 | 1.68% | 0.00% | 0.00% | 0.00% | 0.84% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 72.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 72.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 72.1% | 0.00% [0.00%, 0.00%] | -0.00% [-0.00%, 0.00%] | 0.00 |
| 5% | 74.0% | 0.00% [0.00%, 2.00%] | 0.27% [0.00%, 0.68%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/WA/Huawei, M1/group/jpeg75_resize05/Apple, M1/group/jpeg75_resize05/Huawei, M1/group/jpeg75_resize05/other, M1/group/native/Huawei, M1/group/screenshot_sim/Huawei, M1/group/screenshot_sim/Samsung, M1/group/screenshot_sim/other, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/WA/Huawei, pooled@0.01/group/jpeg75_resize05/Huawei, pooled@0.01/group/native/Huawei, pooled@0.01/group/screenshot_sim/Huawei)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 79.5% [78.9%, 80.6%] | 20.4% [19.6%, 21.8%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 89.4% [89.1%, 89.8%] | 32.6% [32.1%, 33.4%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 94.3% [94.2%, 94.5%] | 42.9% [42.4%, 43.5%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 98.1% [98.1%, 98.2%] | 58.2% [57.9%, 58.5%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 72.1% [71.7%, 72.8%] | 13.5% [13.1%, 14.2%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 77.3% [76.6%, 78.5%] | 18.3% [17.4%, 20.0%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 87.5% [86.9%, 88.3%] | 30.7% [29.8%, 31.9%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 96.8% [96.7%, 96.9%] | 51.1% [50.6%, 51.7%] |
| M1 | 2.29% [0.84%, 4.28%] | 3.91% [1.69%, 7.02%] | 93.0% [93.0%, 93.0%] | 40.2% [40.2%, 40.2%] |

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
| 0.5% | 72.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 77.3% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 87.5% | 2.32% [0.65%, 3.82%] | 0.52% [0.11%, 0.95%] | 0.00 |
| 5% | 96.8% | -1.40% [-2.54%, 0.17%] | -0.36% [-0.70%, -0.08%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 3.51% [0.88%, 14.29%] | 57.6% [57.4%, 57.8%] | 30.2% [29.7%, 30.8%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 5.75% [1.18%, 15.52%] | 60.7% [60.3%, 61.0%] | 37.8% [37.2%, 38.6%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 11.49% [4.65%, 23.26%] | 68.1% [68.0%, 68.2%] | 52.9% [52.7%, 53.1%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 21.84% [13.51%, 35.71%] | 72.8% [72.7%, 72.9%] | 65.9% [65.7%, 66.1%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 1.75% [0.00%, 7.14%] | 53.2% [53.0%, 53.5%] | 19.6% [19.2%, 20.2%] |
| wg@0.01 | 0.17% [0.00%, 0.34%] | 1.75% [0.00%, 7.14%] | 53.2% [53.0%, 53.5%] | 19.6% [19.2%, 20.2%] |
| wg@0.02 | 0.28% [0.00%, 0.67%] | 3.45% [0.00%, 10.34%] | 53.5% [53.2%, 53.9%] | 20.4% [19.9%, 21.3%] |
| wg@0.05 | 1.06% [0.44%, 1.79%] | 10.34% [4.22%, 21.43%] | 61.1% [61.1%, 61.2%] | 38.5% [38.4%, 38.7%] |
| M1 | 0.50% [0.06%, 1.06%] | 3.51% [0.88%, 14.29%] | 59.8% [59.8%, 59.8%] | 36.3% [36.2%, 36.3%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 1.0, "0.01": 0.57, "0.02": 0.04, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.96% | 0.00% | 0.56% | 0.00% | 0.00% |
| pooled@0.01 | 2.51% | 0.00% | 0.56% | 0.00% | 0.28% |
| wg@0.01 | 0.84% | 0.00% | 0.00% | 0.00% | 0.00% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 53.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 53.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 53.5% | 1.69% [0.00%, 6.90%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 5% | 61.1% | 4.07% [0.00%, 9.66%] | 0.32% [0.07%, 0.70%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/group/native/Huawei, M1/group/native/Samsung, pooled@0.01/platform/native, pooled@0.01/group/native/Huawei, pooled@0.01/group/native/Samsung)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 57.6% [57.4%, 57.8%] | 30.2% [29.7%, 30.8%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 60.7% [60.3%, 61.0%] | 37.8% [37.2%, 38.6%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 68.1% [68.0%, 68.2%] | 52.9% [52.7%, 53.1%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 72.8% [72.7%, 72.9%] | 65.9% [65.7%, 66.1%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 53.2% [53.0%, 53.5%] | 19.6% [19.2%, 20.2%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 58.3% [58.1%, 58.5%] | 30.6% [30.1%, 31.2%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 61.5% [61.2%, 62.0%] | 39.4% [38.8%, 40.5%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 69.8% [69.7%, 69.9%] | 57.7% [57.6%, 58.0%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 59.8% [59.8%, 59.8%] | 36.3% [36.2%, 36.3%] |

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
| 0.5% | 53.2% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 58.3% | 0.14% [0.00%, 0.44%] | 0.00% [-0.08%, 0.08%] | 0.00 |
| 2% | 61.5% | 1.98% [0.66%, 3.67%] | 0.33% [0.08%, 0.66%] | 0.00 |
| 5% | 69.8% | 0.38% [-0.28%, 1.28%] | 0.01% [-0.19%, 0.21%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
