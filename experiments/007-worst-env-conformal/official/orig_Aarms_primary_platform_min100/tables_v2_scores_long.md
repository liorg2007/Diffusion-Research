# H2-v2 tables (scores_long)

Amendment 3 item 4. LODO x LOPO cells (device x platform held out), per-group split-conformal thresholds, 95% CIs by resampling the 25 test devices. Flag fake iff score > threshold. Rule definitions and the grouping rule: h2v2.py docstring / notes.md.

Brand derivable from src_path: **True**. Groupings: platform.

## A0 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.79% [0.02%, 2.07%] | 1.92% [0.12%, 4.83%] | 23.1% [22.5%, 24.1%] | 15.5% [14.9%, 16.4%] |
| pooled@0.01 | 1.40% [0.06%, 3.77%] | 2.43% [0.24%, 6.21%] | 29.1% [28.2%, 30.7%] | 23.2% [22.3%, 24.6%] |
| pooled@0.02 | 2.19% [0.30%, 5.48%] | 3.15% [0.60%, 7.92%] | 44.5% [43.6%, 46.1%] | 36.4% [35.4%, 38.2%] |
| pooled@0.05 | 5.60% [2.67%, 9.92%] | 7.78% [4.14%, 13.31%] | 68.2% [67.8%, 68.9%] | 62.1% [61.4%, 63.0%] |
| wg@0.005 | 0.25% [0.00%, 0.67%] | 0.84% [0.00%, 2.08%] | 12.1% [11.8%, 12.8%] | 7.3% [7.1%, 7.7%] |
| wg@0.01 | 0.85% [0.03%, 2.29%] | 2.15% [0.16%, 5.43%] | 22.5% [21.9%, 23.7%] | 14.9% [14.2%, 16.1%] |
| wg@0.02 | 1.72% [0.10%, 4.66%] | 2.91% [0.32%, 7.40%] | 32.5% [31.2%, 34.6%] | 26.3% [25.2%, 28.3%] |
| wg@0.05 | 4.35% [1.90%, 8.18%] | 5.71% [2.91%, 10.84%] | 63.8% [63.3%, 64.6%] | 56.9% [56.4%, 57.7%] |
| M1 | 1.68% [0.31%, 4.00%] | 2.51% [0.63%, 6.13%] | 42.9% [42.9%, 42.9%] | 35.5% [35.5%, 35.5%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.24% | 2.51% | 1.80% | 2.15% | 0.72% |
| pooled@0.01 | 0.72% | 2.43% | 1.48% | 1.52% | 0.88% |
| wg@0.01 | 0.44% | 2.15% | 0.48% | 1.00% | 0.20% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 12.1% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 22.5% | 0.25% [0.01%, 0.60%] | 0.15% [-0.00%, 0.34%] | 0.00 |
| 2% | 32.5% | 0.17% [0.00%, 0.36%] | 0.09% [-0.04%, 0.28%] | 0.00 |
| 5% | 63.8% | -0.22% [-0.52%, -0.05%] | -0.15% [-0.27%, -0.04%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/jpeg75_resize05, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## A1 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.69% [0.19%, 1.53%] | 1.68% [0.40%, 3.91%] | 32.8% [32.6%, 33.3%] | 26.6% [26.2%, 27.3%] |
| pooled@0.01 | 1.31% [0.33%, 3.03%] | 2.08% [0.63%, 4.82%] | 39.2% [38.6%, 40.4%] | 33.7% [33.3%, 34.6%] |
| pooled@0.02 | 2.41% [0.65%, 5.44%] | 3.11% [1.12%, 7.36%] | 51.8% [51.0%, 53.1%] | 43.0% [42.3%, 44.3%] |
| pooled@0.05 | 5.53% [2.78%, 9.60%] | 8.22% [4.75%, 12.87%] | 73.1% [72.7%, 73.8%] | 68.5% [68.0%, 69.3%] |
| wg@0.005 | 0.18% [0.04%, 0.38%] | 0.56% [0.12%, 1.09%] | 21.6% [21.4%, 21.9%] | 16.2% [16.1%, 16.5%] |
| wg@0.01 | 0.77% [0.19%, 1.77%] | 1.76% [0.40%, 4.11%] | 33.5% [33.1%, 34.1%] | 26.7% [26.1%, 27.5%] |
| wg@0.02 | 1.71% [0.42%, 3.99%] | 2.43% [0.71%, 5.67%] | 43.8% [43.0%, 45.1%] | 37.5% [36.9%, 38.5%] |
| wg@0.05 | 4.47% [2.00%, 8.22%] | 6.11% [3.08%, 10.62%] | 68.6% [68.2%, 69.2%] | 62.3% [61.7%, 63.2%] |
| M1 | 2.04% [0.77%, 4.08%] | 2.55% [1.23%, 5.55%] | 52.3% [52.3%, 52.3%] | 44.4% [44.4%, 44.4%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 1.60% | 2.55% | 2.11% | 2.51% | 1.44% |
| pooled@0.01 | 0.72% | 2.08% | 1.20% | 1.64% | 0.92% |
| wg@0.01 | 0.56% | 1.76% | 0.44% | 0.68% | 0.40% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 21.6% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 33.5% | 0.04% [-0.17%, 0.09%] | 0.03% [-0.04%, 0.11%] | 0.00 |
| 2% | 43.8% | 0.05% [-0.14%, 0.19%] | 0.01% [-0.06%, 0.10%] | 0.00 |
| 5% | 68.6% | 0.13% [-0.34%, 0.34%] | -0.08% [-0.20%, 0.02%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, pooled@0.01/platform/WA, pooled@0.01/group/WA)

## A2 / grouping platform  [LOGO: logo_<g> scores reals and generator g's fakes]

Held-out groups (5, min size 358 reals): FBH (n=358, 25 dev); WA (n=358, 25 dev); native (n=358, 25 dev)

| rule | FPR pooled [CI] | worst-group FPR [CI] | TPR clean [CI] | TPR jpeg75_resize05 [CI] |
|---|---|---|---|---|
| pooled@0.005 | 0.45% [0.12%, 0.86%] | 1.04% [0.32%, 1.96%] | 21.8% [21.6%, 22.1%] | 14.9% [14.6%, 15.3%] |
| pooled@0.01 | 1.11% [0.33%, 2.21%] | 1.60% [0.59%, 3.14%] | 28.8% [28.4%, 29.4%] | 24.3% [23.7%, 25.1%] |
| pooled@0.02 | 2.30% [0.69%, 4.69%] | 3.23% [1.08%, 6.35%] | 39.7% [39.1%, 40.7%] | 35.7% [35.3%, 36.5%] |
| pooled@0.05 | 5.12% [2.38%, 9.14%] | 6.23% [3.21%, 10.44%] | 64.2% [63.6%, 65.0%] | 57.6% [56.9%, 58.6%] |
| wg@0.005 | 0.16% [0.04%, 0.31%] | 0.56% [0.12%, 1.15%] | 10.0% [9.7%, 10.4%] | 5.8% [5.6%, 6.1%] |
| wg@0.01 | 0.57% [0.15%, 1.09%] | 1.36% [0.36%, 2.53%] | 22.9% [22.6%, 23.3%] | 15.3% [14.8%, 16.0%] |
| wg@0.02 | 1.63% [0.42%, 3.46%] | 2.08% [0.72%, 4.48%] | 34.6% [34.0%, 35.4%] | 30.2% [29.4%, 31.0%] |
| wg@0.05 | 4.36% [1.80%, 8.24%] | 5.59% [2.49%, 9.91%] | 58.1% [57.3%, 59.1%] | 50.7% [49.8%, 52.0%] |
| M1 | 5.71% [3.19%, 9.25%] | 6.54% [4.09%, 10.45%] | 64.0% [64.0%, 64.0%] | 62.0% [62.0%, 62.0%] |

Share of calibration groups / pools with no finite conformal threshold (clipped to the max): worst-group {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}; pooled {"0.005": 0.0, "0.01": 0.0, "0.02": 0.0, "0.05": 0.0}

### Held-out platform FPR (LOPO view, pooled over devices)

| rule | native | WA | FBH | jpeg75_resize05 | screenshot_sim |
|---|---|---|---|---|---|
| M1 | 4.35% | 6.23% | 6.50% | 6.54% | 4.95% |
| pooled@0.01 | 0.60% | 1.60% | 1.16% | 1.08% | 1.12% |
| wg@0.01 | 0.24% | 1.36% | 0.36% | 0.56% | 0.32% |

### Worst-group vs pooled at MATCHED clean TPR (diff = FPR(wg) - FPR(pooled interpolated at the same TPR); <0 favours worst-group)

| alpha | wg TPR | worst-group-FPR diff [CI] | pooled-FPR diff [CI] | clamped (boot share) |
|---|---|---|---|---|
| 0.5% | 10.0% | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 0.00 |
| 1% | 22.9% | 0.24% [-0.01%, 0.49%] | 0.02% [-0.01%, 0.06%] | 0.00 |
| 2% | 34.6% | -0.13% [-0.26%, 0.09%] | -0.02% [-0.09%, 0.06%] | 0.00 |
| 5% | 58.1% | 0.02% [-0.31%, 0.17%] | -0.01% [-0.13%, 0.12%] | 0.00 |

**H2-v2 verdict (proposed rule): NOT_SUPPORTED**; problem real (M1 or M2 > 2% on a held-out platform/group): True (M1/platform/native, M1/platform/WA, M1/platform/FBH, M1/platform/jpeg75_resize05, M1/platform/screenshot_sim, M1/group/FBH, M1/group/WA, M1/group/jpeg75_resize05, M1/group/native, M1/group/screenshot_sim)
