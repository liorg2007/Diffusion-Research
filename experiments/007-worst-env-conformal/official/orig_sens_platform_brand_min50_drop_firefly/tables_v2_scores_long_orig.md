# H2-v2 tables (scores_long_orig)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform_brand, platform.

## B0 / grouping platform_brand  [single model 'full' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 5.22% [0.00%, 15.07%] | 26.6% [25.9%, 27.5%] | 20.5% [20.0%, 21.3%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 6.09% [0.86%, 18.05%] | 35.8% [35.2%, 37.0%] | 27.8% [27.2%, 28.9%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 7.83% [2.33%, 23.76%] | 47.8% [46.8%, 49.6%] | 38.1% [37.1%, 39.9%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 10.43% [7.02%, 25.59%] | 76.6% [76.2%, 77.1%] | 66.3% [65.9%, 67.0%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 2.61% [0.00%, 6.82%] | 14.5% [14.0%, 15.3%] | 8.5% [8.1%, 9.1%] |
| wg@0.01 | 0.34% [0.00%, 0.89%] | 2.61% [0.00%, 6.82%] | 14.5% [14.0%, 15.3%] | 8.5% [8.1%, 9.1%] |
| wg@0.02 | 0.34% [0.00%, 0.89%] | 2.61% [0.00%, 6.82%] | 23.3% [23.1%, 23.5%] | 15.9% [15.6%, 16.2%] |
| wg@0.05 | 1.79% [0.11%, 4.88%] | 6.96% [1.37%, 20.83%] | 39.4% [38.5%, 41.2%] | 31.1% [30.3%, 32.7%] |
| M1 | 1.34% [0.11%, 3.38%] | 5.22% [1.39%, 15.07%] | 42.0% [42.0%, 42.0%] | 34.0% [34.0%, 34.0%] |

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
| 0.5% | 14.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 14.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 23.3% | -1.33% [-4.36%, 0.00%] | -0.09% [-0.25%, 0.00%] | 0.00 |
| 5% | 39.4% | -0.22% [-0.80%, 1.48%] | -0.02% [-0.18%, 0.14%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/FBH/Samsung, M1/group/FBH/other, M1/group/WA/other, M1/group/jpeg75_resize05/other, M1/group/native/other, pooled@0.01/platform/WA, pooled@0.01/group/FBH/other, pooled@0.01/group/WA/other, pooled@0.01/group/jpeg75_resize05/other, pooled@0.01/group/screenshot_sim/other)

## B0 / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.78% [0.00%, 2.13%] | 1.96% [0.00%, 5.26%] | 26.6% [25.9%, 27.5%] | 20.5% [20.0%, 21.3%] |
| pooled@0.01 | 1.28% [0.06%, 3.39%] | 2.51% [0.28%, 6.37%] | 35.8% [35.2%, 37.0%] | 27.8% [27.2%, 28.9%] |
| pooled@0.02 | 2.18% [0.28%, 5.46%] | 3.35% [0.56%, 8.36%] | 47.8% [46.8%, 49.6%] | 38.1% [37.1%, 39.9%] |
| pooled@0.05 | 5.47% [2.64%, 9.72%] | 7.26% [3.91%, 12.26%] | 76.6% [76.2%, 77.1%] | 66.3% [65.9%, 67.0%] |
| wg@0.005 | 0.34% [0.00%, 0.89%] | 1.12% [0.00%, 2.51%] | 14.5% [14.0%, 15.3%] | 8.5% [8.1%, 9.1%] |
| wg@0.01 | 0.78% [0.00%, 2.22%] | 1.96% [0.00%, 5.26%] | 25.7% [25.1%, 26.8%] | 19.9% [19.4%, 20.8%] |
| wg@0.02 | 1.79% [0.06%, 4.89%] | 3.07% [0.28%, 8.03%] | 37.5% [36.3%, 39.5%] | 29.7% [28.6%, 31.5%] |
| wg@0.05 | 4.25% [1.75%, 8.17%] | 5.31% [2.81%, 10.31%] | 70.4% [69.5%, 71.4%] | 60.2% [59.3%, 61.3%] |
| M1 | 1.34% [0.11%, 3.38%] | 2.23% [0.28%, 5.54%] | 42.0% [42.0%, 42.0%] | 34.0% [34.0%, 34.0%] |

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
| 0.5% | 14.5% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 25.7% | 0.10% [0.00%, 0.17%] | 0.13% [-0.08%, 0.30%] | 0.00 |
| 2% | 37.5% | 0.36% [0.00%, 0.99%] | 0.19% [0.00%, 0.40%] | 0.00 |
| 5% | 70.4% | -0.28% [-0.79%, 0.36%] | 0.01% [-0.17%, 0.18%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/group/WA, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## B0_twinfree / grouping platform_brand  [single model 'full' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 3.48% [0.00%, 9.59%] | 22.3% [21.9%, 22.9%] | 14.2% [13.7%, 15.1%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 6.96% [0.00%, 20.83%] | 28.7% [27.3%, 31.3%] | 21.8% [20.5%, 24.1%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 7.83% [1.75%, 23.76%] | 49.8% [48.7%, 51.7%] | 40.1% [39.0%, 41.9%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 11.30% [7.02%, 26.22%] | 70.9% [70.5%, 71.4%] | 61.4% [61.1%, 62.0%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 1.15% [0.00%, 3.45%] | 14.7% [14.6%, 14.9%] | 6.9% [6.7%, 7.1%] |
| wg@0.01 | 0.22% [0.00%, 0.56%] | 1.15% [0.00%, 3.45%] | 14.7% [14.6%, 14.9%] | 6.9% [6.7%, 7.1%] |
| wg@0.02 | 0.28% [0.00%, 0.72%] | 1.15% [0.00%, 3.45%] | 16.5% [16.2%, 17.1%] | 8.0% [7.7%, 8.5%] |
| wg@0.05 | 2.07% [0.11%, 5.65%] | 7.83% [0.88%, 23.76%] | 36.7% [35.4%, 39.4%] | 30.0% [28.8%, 32.3%] |
| M1 | 2.29% [0.56%, 5.17%] | 6.96% [2.35%, 20.83%] | 56.2% [56.2%, 56.2%] | 44.9% [44.9%, 44.9%] |

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
| 0.5% | 14.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 14.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 16.5% | -0.62% [-1.54%, 0.00%] | -0.02% [-0.11%, 0.08%] | 0.00 |
| 5% | 36.7% | 0.87% [0.00%, 2.94%] | 0.06% [-0.17%, 0.37%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH/Samsung, M1/group/FBH/other, M1/group/WA/other, M1/group/jpeg75_resize05/Apple, M1/group/jpeg75_resize05/Samsung, M1/group/jpeg75_resize05/other, M1/group/native/Samsung, M1/group/native/other, M1/group/screenshot_sim/Samsung, M1/group/screenshot_sim/other, pooled@0.01/platform/WA, pooled@0.01/group/FBH/other, pooled@0.01/group/WA/other, pooled@0.01/group/jpeg75_resize05/other, pooled@0.01/group/screenshot_sim/other)

## B0_twinfree / grouping platform  [single model 'full' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.89% [0.00%, 2.35%] | 1.40% [0.00%, 3.36%] | 22.3% [21.9%, 22.9%] | 14.2% [13.7%, 15.1%] |
| pooled@0.01 | 1.56% [0.00%, 4.36%] | 2.51% [0.00%, 6.93%] | 28.7% [27.3%, 31.3%] | 21.8% [20.5%, 24.1%] |
| pooled@0.02 | 2.51% [0.34%, 6.30%] | 3.07% [0.56%, 8.03%] | 49.8% [48.7%, 51.7%] | 40.1% [39.0%, 41.9%] |
| pooled@0.05 | 5.08% [2.08%, 9.39%] | 7.54% [4.18%, 12.71%] | 70.9% [70.5%, 71.4%] | 61.4% [61.1%, 62.0%] |
| wg@0.005 | 0.22% [0.00%, 0.56%] | 0.56% [0.00%, 1.39%] | 14.7% [14.6%, 14.9%] | 6.9% [6.7%, 7.1%] |
| wg@0.01 | 1.17% [0.00%, 3.24%] | 1.96% [0.00%, 5.26%] | 23.6% [22.9%, 25.1%] | 15.4% [14.7%, 17.0%] |
| wg@0.02 | 1.84% [0.11%, 4.99%] | 2.51% [0.28%, 6.93%] | 34.4% [33.0%, 37.1%] | 28.1% [27.0%, 30.2%] |
| wg@0.05 | 4.13% [1.56%, 8.12%] | 6.70% [3.33%, 11.95%] | 67.8% [67.5%, 68.3%] | 58.0% [57.7%, 58.5%] |
| M1 | 2.29% [0.56%, 5.17%] | 3.07% [1.12%, 7.20%] | 56.2% [56.2%, 56.2%] | 44.9% [44.9%, 44.9%] |

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
| 0.5% | 14.7% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 23.6% | 0.28% [0.00%, 0.45%] | 0.17% [-0.17%, 0.45%] | 0.00 |
| 2% | 34.4% | 0.00% [-0.16%, 0.00%] | -0.05% [-0.18%, 0.08%] | 0.00 |
| 5% | 67.8% | 0.11% [-0.81%, 0.34%] | -0.23% [-0.50%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## bfree / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 8.77% [2.38%, 14.29%] | 62.2% [61.9%, 62.6%] | 9.8% [9.7%, 10.0%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 10.53% [3.51%, 14.29%] | 68.8% [68.5%, 69.2%] | 13.1% [12.9%, 13.3%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 12.28% [4.76%, 21.43%] | 78.5% [78.3%, 78.8%] | 22.0% [21.6%, 22.6%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 15.79% [8.53%, 42.86%] | 86.9% [86.7%, 87.1%] | 38.2% [38.0%, 38.4%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 7.02% [0.00%, 14.29%] | 35.0% [34.0%, 37.1%] | 4.1% [3.9%, 4.5%] |
| wg@0.01 | 0.34% [0.00%, 0.68%] | 7.02% [0.00%, 14.29%] | 35.0% [34.0%, 37.1%] | 4.1% [3.9%, 4.5%] |
| wg@0.02 | 0.34% [0.00%, 0.68%] | 7.02% [0.00%, 14.29%] | 35.0% [34.0%, 37.1%] | 4.1% [3.9%, 4.5%] |
| wg@0.05 | 0.56% [0.11%, 1.06%] | 10.53% [2.82%, 14.29%] | 59.5% [59.1%, 60.1%] | 8.9% [8.7%, 9.1%] |
| M1 | 3.80% [2.23%, 5.53%] | 14.04% [7.14%, 35.71%] | 85.7% [85.7%, 85.7%] | 33.7% [33.7%, 33.7%] |

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
| 0.5% | 35.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 35.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 35.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 5% | 59.5% | 3.41% [0.00%, 10.71%] | 0.16% [-0.11%, 0.42%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH/Apple, M1/group/FBH/Huawei, M1/group/WA/Apple, M1/group/WA/Huawei, M1/group/WA/other, M1/group/jpeg75_resize05/Huawei, M1/group/jpeg75_resize05/Samsung, M1/group/jpeg75_resize05/other, M1/group/native/Huawei, M1/group/screenshot_sim/Apple, M1/group/screenshot_sim/Huawei, M1/group/screenshot_sim/other, pooled@0.01/platform/WA, pooled@0.01/group/WA/Apple, pooled@0.01/group/WA/Huawei, pooled@0.01/group/screenshot_sim/Huawei)

## bfree / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.61% [0.22%, 1.11%] | 2.23% [0.56%, 4.20%] | 62.2% [61.9%, 62.6%] | 9.8% [9.7%, 10.0%] |
| pooled@0.01 | 0.95% [0.39%, 1.62%] | 3.07% [1.38%, 5.25%] | 68.8% [68.5%, 69.2%] | 13.1% [12.9%, 13.3%] |
| pooled@0.02 | 1.84% [0.90%, 2.92%] | 3.91% [2.21%, 6.44%] | 78.5% [78.3%, 78.8%] | 22.0% [21.6%, 22.6%] |
| pooled@0.05 | 5.42% [3.13%, 8.02%] | 6.42% [4.20%, 10.45%] | 86.9% [86.7%, 87.1%] | 38.2% [38.0%, 38.4%] |
| wg@0.005 | 0.34% [0.00%, 0.68%] | 1.68% [0.00%, 3.38%] | 35.0% [34.0%, 37.1%] | 4.1% [3.9%, 4.5%] |
| wg@0.01 | 0.61% [0.22%, 1.01%] | 2.79% [0.84%, 4.97%] | 61.8% [61.6%, 62.0%] | 10.0% [9.9%, 10.1%] |
| wg@0.02 | 0.89% [0.34%, 1.56%] | 3.07% [1.12%, 5.25%] | 66.8% [66.0%, 67.6%] | 13.0% [12.6%, 13.3%] |
| wg@0.05 | 4.86% [2.75%, 7.34%] | 5.31% [3.92%, 9.01%] | 86.3% [86.1%, 86.4%] | 36.6% [36.2%, 37.0%] |
| M1 | 3.80% [2.23%, 5.53%] | 4.47% [3.34%, 7.56%] | 85.7% [85.7%, 85.7%] | 33.7% [33.7%, 33.7%] |

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
| 0.5% | 35.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 61.8% | 0.56% [0.00%, 1.39%] | 0.00% [-0.17%, 0.17%] | 0.00 |
| 2% | 66.8% | 0.21% [0.00%, 0.69%] | 0.03% [-0.04%, 0.13%] | 0.00 |
| 5% | 86.3% | -0.85% [-1.98%, 0.00%] | -0.23% [-0.51%, -0.01%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## clipdet / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 26.32% [15.25%, 57.14%] | 11.9% [11.7%, 12.1%] | 16.3% [16.0%, 16.6%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 40.35% [27.72%, 71.43%] | 17.5% [17.4%, 17.7%] | 24.1% [23.9%, 24.2%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 40.35% [31.40%, 71.43%] | 23.4% [23.0%, 23.8%] | 31.3% [30.9%, 31.8%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 56.14% [52.33%, 72.41%] | 40.5% [40.3%, 40.7%] | 52.0% [51.8%, 52.3%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 6.09% [3.79%, 11.30%] | 6.3% [6.2%, 6.3%] | 7.8% [7.8%, 7.8%] |
| wg@0.01 | 1.06% [0.50%, 1.79%] | 6.09% [3.79%, 11.30%] | 6.3% [6.2%, 6.3%] | 7.8% [7.8%, 7.8%] |
| wg@0.02 | 1.51% [0.56%, 2.87%] | 10.43% [4.65%, 21.43%] | 7.1% [6.8%, 7.5%] | 8.8% [8.3%, 9.4%] |
| wg@0.05 | 6.59% [4.57%, 8.86%] | 36.84% [25.86%, 64.29%] | 16.6% [16.5%, 16.8%] | 22.6% [22.4%, 22.8%] |
| M1 | 11.12% [8.63%, 13.87%] | 49.12% [38.89%, 71.43%] | 28.8% [28.8%, 28.8%] | 38.2% [38.2%, 38.2%] |

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
| 0.5% | 6.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 6.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 7.1% | -2.47% [-5.57%, -0.59%] | -0.38% [-0.60%, -0.19%] | 0.00 |
| 5% | 16.6% | -2.74% [-5.79%, 0.00%] | -0.59% [-0.91%, -0.31%] | 0.00 |

**H2-v2 verdict (proposed rule): INCONCLUSIVE**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH/Huawei, M1/group/FBH/other, M1/group/WA/other, M1/group/jpeg75_resize05/Apple, M1/group/jpeg75_resize05/Huawei, M1/group/jpeg75_resize05/Samsung, M1/group/jpeg75_resize05/other, M1/group/native/Huawei, M1/group/native/Samsung, M1/group/native/other, M1/group/screenshot_sim/Apple, M1/group/screenshot_sim/Huawei, M1/group/screenshot_sim/Samsung, M1/group/screenshot_sim/other, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05/Apple, pooled@0.01/group/jpeg75_resize05/Huawei, pooled@0.01/group/jpeg75_resize05/Samsung, pooled@0.01/group/jpeg75_resize05/other, pooled@0.01/group/native/Huawei, pooled@0.01/group/native/other, pooled@0.01/group/screenshot_sim/Apple, pooled@0.01/group/screenshot_sim/Huawei, pooled@0.01/group/screenshot_sim/Samsung, pooled@0.01/group/screenshot_sim/other)

## clipdet / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 4.19% [2.41%, 6.31%] | 15.64% [8.94%, 22.75%] | 11.9% [11.7%, 12.1%] | 16.3% [16.0%, 16.6%] |
| pooled@0.01 | 7.43% [5.21%, 9.92%] | 27.37% [18.94%, 36.21%] | 17.5% [17.4%, 17.7%] | 24.1% [23.9%, 24.2%] |
| pooled@0.02 | 9.44% [6.93%, 12.36%] | 31.84% [22.99%, 40.90%] | 23.4% [23.0%, 23.8%] | 31.3% [30.9%, 31.8%] |
| pooled@0.05 | 17.49% [14.02%, 21.24%] | 51.96% [44.04%, 60.22%] | 40.5% [40.3%, 40.7%] | 52.0% [51.8%, 52.3%] |
| wg@0.005 | 1.06% [0.50%, 1.79%] | 3.35% [1.66%, 5.37%] | 6.3% [6.2%, 6.3%] | 7.8% [7.8%, 7.8%] |
| wg@0.01 | 3.63% [1.92%, 5.84%] | 12.57% [6.76%, 18.82%] | 11.8% [11.5%, 12.2%] | 15.8% [15.5%, 16.5%] |
| wg@0.02 | 7.21% [4.92%, 9.78%] | 25.70% [17.60%, 33.98%] | 18.2% [18.0%, 18.5%] | 24.6% [24.3%, 25.0%] |
| wg@0.05 | 12.57% [9.69%, 15.71%] | 39.39% [30.53%, 48.73%] | 31.7% [31.4%, 32.2%] | 42.4% [41.9%, 43.1%] |
| M1 | 11.12% [8.63%, 13.87%] | 38.27% [29.41%, 47.88%] | 28.8% [28.8%, 28.8%] | 38.2% [38.2%, 38.2%] |

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
| 0.5% | 6.3% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 11.8% | -2.42% [-4.54%, -0.90%] | -0.40% [-0.80%, -0.12%] | 0.00 |
| 2% | 18.2% | -2.35% [-3.52%, -1.18%] | -0.44% [-0.76%, -0.11%] | 0.00 |
| 5% | 31.7% | -2.05% [-3.53%, -0.76%] | -0.42% [-0.83%, -0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/jpeg75_resize05, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/jpeg75_resize05, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## communityforensics / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 10.53% [0.00%, 35.71%] | 79.2% [78.6%, 80.4%] | 15.9% [15.1%, 17.2%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 12.28% [1.39%, 42.86%] | 89.7% [89.5%, 90.1%] | 27.2% [26.8%, 28.0%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 19.30% [4.85%, 50.00%] | 94.4% [94.3%, 94.6%] | 37.1% [36.6%, 37.7%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 24.56% [12.00%, 64.29%] | 98.1% [98.0%, 98.2%] | 53.3% [53.0%, 53.6%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 10.53% [0.00%, 35.71%] | 71.4% [71.0%, 72.2%] | 9.3% [9.0%, 10.0%] |
| wg@0.01 | 0.50% [0.00%, 1.42%] | 10.53% [0.00%, 35.71%] | 71.4% [71.0%, 72.2%] | 9.3% [9.0%, 10.0%] |
| wg@0.02 | 0.50% [0.00%, 1.42%] | 10.53% [0.00%, 35.71%] | 71.4% [71.0%, 72.2%] | 9.3% [9.0%, 10.0%] |
| wg@0.05 | 1.01% [0.00%, 2.84%] | 10.53% [0.00%, 35.71%] | 73.3% [72.5%, 74.9%] | 12.0% [11.1%, 13.8%] |
| M1 | 2.29% [0.84%, 4.28%] | 15.79% [4.23%, 35.71%] | 93.4% [93.4%, 93.4%] | 34.2% [34.2%, 34.2%] |

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
| 0.5% | 71.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 71.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 71.4% | 0.00% [0.00%, 0.00%] | -0.00% [-0.00%, 0.00%] | 0.00 |
| 5% | 73.3% | 0.00% [0.00%, 2.00%] | 0.27% [0.00%, 0.68%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/WA/Huawei, M1/group/jpeg75_resize05/Apple, M1/group/jpeg75_resize05/Huawei, M1/group/jpeg75_resize05/other, M1/group/native/Huawei, M1/group/screenshot_sim/Huawei, M1/group/screenshot_sim/Samsung, M1/group/screenshot_sim/other, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/WA/Huawei, pooled@0.01/group/jpeg75_resize05/Huawei, pooled@0.01/group/native/Huawei, pooled@0.01/group/screenshot_sim/Huawei)

## communityforensics / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.84% [0.00%, 2.36%] | 1.68% [0.00%, 4.66%] | 79.2% [78.6%, 80.4%] | 15.9% [15.1%, 17.2%] |
| pooled@0.01 | 1.40% [0.17%, 3.45%] | 2.51% [0.56%, 5.87%] | 89.7% [89.5%, 90.1%] | 27.2% [26.8%, 28.0%] |
| pooled@0.02 | 3.24% [1.23%, 5.94%] | 5.03% [1.98%, 9.22%] | 94.4% [94.3%, 94.6%] | 37.1% [36.6%, 37.7%] |
| pooled@0.05 | 7.71% [4.61%, 11.45%] | 12.29% [8.22%, 17.37%] | 98.1% [98.0%, 98.2%] | 53.3% [53.0%, 53.6%] |
| wg@0.005 | 0.50% [0.00%, 1.42%] | 1.68% [0.00%, 4.66%] | 71.4% [71.0%, 72.2%] | 9.3% [9.0%, 10.0%] |
| wg@0.01 | 0.89% [0.00%, 2.59%] | 1.68% [0.00%, 4.66%] | 76.8% [76.1%, 78.1%] | 14.1% [13.2%, 15.7%] |
| wg@0.02 | 1.73% [0.34%, 3.95%] | 4.47% [1.41%, 8.12%] | 87.6% [87.1%, 88.4%] | 25.5% [24.6%, 26.7%] |
| wg@0.05 | 5.20% [2.52%, 8.59%] | 7.54% [4.43%, 12.85%] | 96.6% [96.5%, 96.7%] | 45.6% [45.1%, 46.2%] |
| M1 | 2.29% [0.84%, 4.28%] | 3.91% [1.69%, 7.02%] | 93.4% [93.4%, 93.4%] | 34.2% [34.2%, 34.2%] |

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
| 0.5% | 71.4% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 76.8% | 0.00% [0.00%, 0.00%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 2% | 87.6% | 2.33% [0.66%, 3.82%] | 0.52% [0.11%, 0.95%] | 0.00 |
| 5% | 96.6% | -1.30% [-2.35%, 0.20%] | -0.30% [-0.63%, -0.04%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim, pooled@0.01/platform/native, pooled@0.01/platform/screenshot_sim, pooled@0.01/group/native, pooled@0.01/group/screenshot_sim)

## rine / grouping platform_brand  [single model 'pretrained' on all generators]

Held-out groups (20, min size 57 reals): FBH/Apple (n=99, 7 dev); FBH/Huawei (n=57, 4 dev); FBH/Samsung (n=87, 6 dev); FBH/other (n=115, 8 dev); WA/Apple (n=99, 7 dev); WA/Huawei (n=57, 4 dev); WA/Samsung (n=87, 6 dev); WA/other (n=115, 8 dev); native/Apple (n=99, 7 dev); native/Huawei (n=57, 4 dev); native/Samsung (n=87, 6 dev); native/other (n=115, 8 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 3.51% [0.88%, 14.29%] | 60.9% [60.7%, 61.1%] | 30.3% [29.8%, 30.9%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 5.75% [1.18%, 15.52%] | 63.6% [63.3%, 63.9%] | 37.8% [37.2%, 38.6%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 11.49% [4.65%, 23.26%] | 70.7% [70.6%, 70.8%] | 53.5% [53.3%, 53.7%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 21.84% [13.51%, 35.71%] | 74.9% [74.9%, 74.9%] | 66.7% [66.5%, 66.9%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 1.75% [0.00%, 7.14%] | 57.0% [56.8%, 57.3%] | 19.7% [19.4%, 20.4%] |
| wg@0.01 | 0.17% [0.00%, 0.34%] | 1.75% [0.00%, 7.14%] | 57.0% [56.8%, 57.3%] | 19.7% [19.4%, 20.4%] |
| wg@0.02 | 0.28% [0.00%, 0.67%] | 3.45% [0.00%, 10.34%] | 57.2% [57.0%, 57.6%] | 20.6% [20.0%, 21.4%] |
| wg@0.05 | 1.06% [0.44%, 1.79%] | 10.34% [4.22%, 21.43%] | 64.2% [64.1%, 64.2%] | 38.7% [38.6%, 38.9%] |
| M1 | 0.50% [0.06%, 1.06%] | 3.51% [0.88%, 14.29%] | 62.9% [62.9%, 62.9%] | 36.2% [36.2%, 36.2%] |

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
| 0.5% | 57.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 57.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 2% | 57.2% | 1.69% [0.00%, 6.90%] | 0.11% [0.00%, 0.34%] | 0.00 |
| 5% | 64.2% | 3.87% [0.00%, 9.26%] | 0.30% [0.05%, 0.65%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/group/native/Huawei, M1/group/native/Samsung, pooled@0.01/platform/native, pooled@0.01/group/native/Huawei, pooled@0.01/group/native/Samsung)

## rine / grouping platform  [single model 'pretrained' on all generators]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.06%, 1.00%] | 1.96% [0.28%, 4.16%] | 60.9% [60.7%, 61.1%] | 30.3% [29.8%, 30.9%] |
| pooled@0.01 | 0.67% [0.11%, 1.39%] | 2.51% [0.56%, 5.26%] | 63.6% [63.3%, 63.9%] | 37.8% [37.2%, 38.6%] |
| pooled@0.02 | 1.68% [0.79%, 2.75%] | 5.59% [2.77%, 9.19%] | 70.7% [70.6%, 70.8%] | 53.5% [53.3%, 53.7%] |
| pooled@0.05 | 5.20% [3.42%, 7.04%] | 11.73% [6.78%, 17.14%] | 74.9% [74.9%, 74.9%] | 66.7% [66.5%, 66.9%] |
| wg@0.005 | 0.17% [0.00%, 0.34%] | 0.84% [0.00%, 1.69%] | 57.0% [56.8%, 57.3%] | 19.7% [19.4%, 20.4%] |
| wg@0.01 | 0.50% [0.11%, 1.05%] | 2.23% [0.56%, 4.44%] | 61.6% [61.5%, 61.8%] | 30.7% [30.3%, 31.3%] |
| wg@0.02 | 1.12% [0.45%, 1.95%] | 5.03% [1.96%, 8.64%] | 64.3% [64.1%, 64.8%] | 39.4% [38.7%, 40.5%] |
| wg@0.05 | 2.51% [1.34%, 3.84%] | 6.98% [3.06%, 11.29%] | 72.2% [72.1%, 72.3%] | 58.3% [58.2%, 58.6%] |
| M1 | 0.50% [0.06%, 1.06%] | 1.96% [0.28%, 4.16%] | 62.9% [62.9%, 62.9%] | 36.2% [36.2%, 36.2%] |

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
| 0.5% | 57.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 61.6% | 0.12% [0.00%, 0.38%] | -0.01% [-0.09%, 0.07%] | 0.00 |
| 2% | 64.3% | 1.97% [0.65%, 3.66%] | 0.32% [0.08%, 0.66%] | 0.00 |
| 5% | 72.2% | 0.45% [-0.23%, 1.38%] | 0.05% [-0.15%, 0.25%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (pooled@0.01/platform/native, pooled@0.01/group/native)
