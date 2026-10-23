# PILOT H2 tables (scores_long)

PILOT (Amendment 2): partial data (VISION + 6 Synthbuster generators), not a Phase 4 verdict. alpha = 1%. Flag fake iff score > threshold. LODO/LOPO details in notes.md.

## A0 (pre-registered primary)  [FPR model: full]

### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)

| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |
|---|---|---|---|---|---|
| M0 | 17.04% | 183/1074 | [14.84%, 19.42%] | 40.00% (D13) | [14.25%, 19.93%] |
| M1 | 0.37% | 4/1074 | [0.10%, 0.95%] | 4.76% (D34) | [0.09%, 0.74%] |
| M2 | 1.02% | 11/1074 | [0.51%, 1.83%] | 7.14% (D34) | [0.37%, 1.86%] |
| M3 | 0.09% | 1/1074 | [0.00%, 0.52%] | 2.38% (D35) | [0.00%, 0.28%] |
| M4 | 0.47% | 5/1074 | [0.15%, 1.08%] | 4.76% (D34) | [0.09%, 0.93%] |

### LOPO FPR (held-out platform; CP 95% CI)

| platform (n) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| native (358) | 5.03% [3.01%, 7.83%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| WA (358) | 29.61% [24.93%, 34.63%] | 1.12% [0.31%, 2.84%] | 3.07% [1.54%, 5.43%] | 1.96% [0.79%, 3.99%] | 2.79% [1.35%, 5.08%] |
| FBH (358) | 16.48% [12.79%, 20.74%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| jpeg75_resize05 (358) | 22.91% [18.65%, 27.61%] | 0.84% [0.17%, 2.43%] | 1.40% [0.46%, 3.23%] | 0.00% [0.00%, 1.03%] | 0.84% [0.17%, 2.43%] |
| screenshot_sim (358) | 11.17% [8.10%, 14.90%] | 0.28% [0.01%, 1.55%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.28% [0.01%, 1.55%] |

Thresholds: per logo_<g> model (see results json).

### LOGO-averaged FPR (mean over logo_<g> models)

| split | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| LODO | 16.79% | 0.54% | 1.07% | 0.09% | 0.51% |
| LOPO native | 5.40% | 0.14% | 0.05% | 0.00% | 0.00% |
| LOPO WA | 29.38% | 1.26% | 3.45% | 1.82% | 2.23% |
| LOPO FBH | 15.60% | 0.23% | 0.33% | 0.00% | 0.00% |
| LOPO jpeg75_resize05 | 23.00% | 1.02% | 1.44% | 0.00% | 0.88% |
| LOPO screenshot_sim | 10.85% | 0.28% | 0.33% | 0.00% | 0.23% |

### TPR (LOGO (model logo_<g> on generator g, thresholds from logo_<g> real scores)); thresholds fit on all 25 test devices' clean reals

| condition | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| clean (macro mean over g) | 96.7% | 73.1% | 77.9% | 46.0% | 68.9% |
| jpeg75_resize05 (macro mean over g) | 98.3% | 77.8% | 82.5% | 48.8% | 74.3% |
| screenshot_sim (macro mean over g) | 96.7% | 73.3% | 76.5% | 46.5% | 69.9% |

TPR ratio M3/M2: clean 0.590, jpeg75_resize05 0.591

Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):

| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |
|---|---|---|---|---|---|---|
| dalle2 | 59.2% | 34.2% | 0.58 | 76.7% | 45.8% | 0.60 |
| dalle3 | 89.2% | 43.3% | 0.49 | 86.7% | 54.2% | 0.62 |
| firefly | 65.0% | 25.0% | 0.38 | 81.7% | 33.3% | 0.41 |
| midjourney-v5 | 79.2% | 50.0% | 0.63 | 81.7% | 45.0% | 0.55 |
| stable-diffusion-2 | 77.5% | 45.8% | 0.59 | 75.8% | 45.0% | 0.59 |
| stable-diffusion-xl | 97.5% | 77.5% | 0.79 | 92.5% | 69.2% | 0.75 |

### PILOT verdict: **INCONCLUSIVE**

- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: True
- M3 LOPO FPR <= 3% on every held-out platform: True
- TPR ratio >= 0.7 (clean 0.590, jpeg75_resize05 0.591): False
- some M0/M1/M2 FPR > 2% on a split: True (LODO/M0, LOPO/native/M0, LOPO/WA/M0, LOPO/WA/M2, LOPO/FBH/M0, LOPO/jpeg75_resize05/M0, LOPO/screenshot_sim/M0)
- fail triggers: M3 LODO > 2%: False; ratio < 0.5: False

- LOGO-averaged check: {"M3_LODO_avg": 0.000931098696461825, "M3_LODO_max_over_models": 0.000931098696461825, "LOPO_M3_avg": {"native": 0.0, "WA": 0.018156424581005585, "FBH": 0.0, "jpeg75_resize05": 0.0, "screenshot_sim": 0.0}, "M3_LODO_avg_le_1p5": true, "LOPO_M3_avg_le_3": true}

## A1  [FPR model: full]

### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)

| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |
|---|---|---|---|---|---|
| M0 | 14.25% | 153/1074 | [12.21%, 16.48%] | 53.33% (D17) | [11.45%, 17.13%] |
| M1 | 1.21% | 13/1074 | [0.65%, 2.06%] | 15.56% (D17) | [0.47%, 2.05%] |
| M2 | 1.02% | 11/1074 | [0.51%, 1.83%] | 15.56% (D17) | [0.37%, 1.77%] |
| M3 | 0.09% | 1/1074 | [0.00%, 0.52%] | 2.38% (D12) | [0.00%, 0.28%] |
| M4 | 0.74% | 8/1074 | [0.32%, 1.46%] | 13.33% (D17) | [0.28%, 1.40%] |

### LOPO FPR (held-out platform; CP 95% CI)

| platform (n) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| native (358) | 6.98% [4.57%, 10.14%] | 0.56% [0.07%, 2.00%] | 0.56% [0.07%, 2.00%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| WA (358) | 22.07% [17.88%, 26.72%] | 2.51% [1.16%, 4.72%] | 3.07% [1.54%, 5.43%] | 1.12% [0.31%, 2.84%] | 1.68% [0.62%, 3.61%] |
| FBH (358) | 13.69% [10.30%, 17.69%] | 0.56% [0.07%, 2.00%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| jpeg75_resize05 (358) | 14.25% [10.79%, 18.30%] | 2.79% [1.35%, 5.08%] | 0.84% [0.17%, 2.43%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| screenshot_sim (358) | 9.78% [6.90%, 13.33%] | 0.28% [0.01%, 1.55%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.28% [0.01%, 1.55%] |

Thresholds: per logo_<g> model (see results json).

### LOGO-averaged FPR (mean over logo_<g> models)

| split | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| LODO | 12.40% | 1.44% | 1.19% | 0.09% | 0.74% |
| LOPO native | 6.19% | 0.79% | 0.42% | 0.00% | 0.00% |
| LOPO WA | 19.32% | 2.61% | 2.89% | 1.16% | 1.63% |
| LOPO FBH | 11.69% | 0.93% | 0.14% | 0.00% | 0.00% |
| LOPO jpeg75_resize05 | 13.50% | 2.61% | 0.98% | 0.00% | 0.23% |
| LOPO screenshot_sim | 9.22% | 0.42% | 0.28% | 0.00% | 0.19% |

### TPR (LOGO (model logo_<g> on generator g, thresholds from logo_<g> real scores)); thresholds fit on all 25 test devices' clean reals

| condition | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| clean (macro mean over g) | 96.2% | 81.7% | 75.1% | 35.0% | 66.2% |
| jpeg75_resize05 (macro mean over g) | 98.1% | 85.8% | 79.4% | 38.9% | 71.8% |
| screenshot_sim (macro mean over g) | 96.4% | 81.8% | 75.4% | 38.6% | 67.4% |

TPR ratio M3/M2: clean 0.466, jpeg75_resize05 0.490

Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):

| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |
|---|---|---|---|---|---|---|
| dalle2 | 55.8% | 15.0% | 0.27 | 65.0% | 18.3% | 0.28 |
| dalle3 | 85.0% | 40.8% | 0.48 | 86.7% | 52.5% | 0.61 |
| firefly | 65.8% | 25.0% | 0.38 | 78.3% | 34.2% | 0.44 |
| midjourney-v5 | 75.8% | 25.0% | 0.33 | 77.5% | 25.8% | 0.33 |
| stable-diffusion-2 | 72.5% | 45.0% | 0.62 | 75.8% | 40.8% | 0.54 |
| stable-diffusion-xl | 95.8% | 59.2% | 0.62 | 93.3% | 61.7% | 0.66 |

### PILOT verdict: **FAIL**

- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: True
- M3 LOPO FPR <= 3% on every held-out platform: True
- TPR ratio >= 0.7 (clean 0.466, jpeg75_resize05 0.490): False
- some M0/M1/M2 FPR > 2% on a split: True (LODO/M0, LOPO/native/M0, LOPO/WA/M0, LOPO/WA/M1, LOPO/WA/M2, LOPO/FBH/M0, LOPO/jpeg75_resize05/M0, LOPO/jpeg75_resize05/M1, LOPO/screenshot_sim/M0)
- fail triggers: M3 LODO > 2%: False; ratio < 0.5: True

- LOGO-averaged check: {"M3_LODO_avg": 0.000931098696461825, "M3_LODO_max_over_models": 0.000931098696461825, "LOPO_M3_avg": {"native": 0.0, "WA": 0.011638733705772812, "FBH": 0.0, "jpeg75_resize05": 0.0, "screenshot_sim": 0.0}, "M3_LODO_avg_le_1p5": true, "LOPO_M3_avg_le_3": true}

## A2  [FPR model: full]

### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)

| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |
|---|---|---|---|---|---|
| M0 | 16.11% | 173/1074 | [13.96%, 18.45%] | 53.33% (D17) | [13.04%, 19.37%] |
| M1 | 3.07% | 33/1074 | [2.12%, 4.29%] | 26.67% (D17) | [1.77%, 4.56%] |
| M2 | 1.40% | 15/1074 | [0.78%, 2.29%] | 20.00% (D17) | [0.47%, 2.42%] |
| M3 | 0.09% | 1/1074 | [0.00%, 0.52%] | 2.38% (D12) | [0.00%, 0.28%] |
| M4 | 0.65% | 7/1074 | [0.26%, 1.34%] | 8.89% (D17) | [0.19%, 1.21%] |

### LOPO FPR (held-out platform; CP 95% CI)

| platform (n) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| native (358) | 13.41% [10.05%, 17.38%] | 1.96% [0.79%, 3.99%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| WA (358) | 18.99% [15.06%, 23.45%] | 3.91% [2.15%, 6.47%] | 1.68% [0.62%, 3.61%] | 1.12% [0.31%, 2.84%] | 1.40% [0.46%, 3.23%] |
| FBH (358) | 15.92% [12.29%, 20.13%] | 3.35% [1.74%, 5.78%] | 0.84% [0.17%, 2.43%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| jpeg75_resize05 (358) | 17.60% [13.80%, 21.95%] | 4.75% [2.79%, 7.49%] | 0.56% [0.07%, 2.00%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| screenshot_sim (358) | 15.64% [12.04%, 19.83%] | 1.96% [0.79%, 3.99%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.28% [0.01%, 1.55%] |

Thresholds: per logo_<g> model (see results json).

### LOGO-averaged FPR (mean over logo_<g> models)

| split | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| LODO | 14.29% | 2.93% | 1.30% | 0.09% | 0.68% |
| LOPO native | 11.55% | 1.91% | 0.14% | 0.00% | 0.00% |
| LOPO WA | 16.90% | 3.45% | 1.77% | 0.88% | 1.26% |
| LOPO FBH | 14.43% | 3.45% | 0.74% | 0.00% | 0.05% |
| LOPO jpeg75_resize05 | 15.64% | 3.72% | 0.70% | 0.00% | 0.00% |
| LOPO screenshot_sim | 13.97% | 1.86% | 0.37% | 0.00% | 0.19% |

### TPR (LOGO (model logo_<g> on generator g, thresholds from logo_<g> real scores)); thresholds fit on all 25 test devices' clean reals

| condition | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| clean (macro mean over g) | 96.0% | 81.4% | 64.6% | 30.0% | 51.5% |
| jpeg75_resize05 (macro mean over g) | 97.4% | 83.3% | 68.1% | 30.1% | 55.0% |
| screenshot_sim (macro mean over g) | 96.0% | 81.2% | 66.2% | 32.2% | 53.1% |

TPR ratio M3/M2: clean 0.465, jpeg75_resize05 0.443

Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):

| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |
|---|---|---|---|---|---|---|
| dalle2 | 42.5% | 11.7% | 0.27 | 44.2% | 15.8% | 0.36 |
| dalle3 | 70.8% | 33.3% | 0.47 | 75.8% | 40.8% | 0.54 |
| firefly | 58.3% | 25.8% | 0.44 | 64.2% | 29.2% | 0.45 |
| midjourney-v5 | 69.2% | 22.5% | 0.33 | 74.2% | 18.3% | 0.25 |
| stable-diffusion-2 | 65.8% | 41.7% | 0.63 | 65.8% | 36.7% | 0.56 |
| stable-diffusion-xl | 80.8% | 45.0% | 0.56 | 84.2% | 40.0% | 0.48 |

### PILOT verdict: **FAIL**

- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: True
- M3 LOPO FPR <= 3% on every held-out platform: True
- TPR ratio >= 0.7 (clean 0.465, jpeg75_resize05 0.443): False
- some M0/M1/M2 FPR > 2% on a split: True (LODO/M0, LODO/M1, LOPO/native/M0, LOPO/WA/M0, LOPO/WA/M1, LOPO/FBH/M0, LOPO/FBH/M1, LOPO/jpeg75_resize05/M0, LOPO/jpeg75_resize05/M1, LOPO/screenshot_sim/M0)
- fail triggers: M3 LODO > 2%: False; ratio < 0.5: True

- LOGO-averaged check: {"M3_LODO_avg": 0.000931098696461825, "M3_LODO_max_over_models": 0.000931098696461825, "LOPO_M3_avg": {"native": 0.0, "WA": 0.008845437616387336, "FBH": 0.0, "jpeg75_resize05": 0.0, "screenshot_sim": 0.0}, "M3_LODO_avg_le_1p5": true, "LOPO_M3_avg_le_3": true}
