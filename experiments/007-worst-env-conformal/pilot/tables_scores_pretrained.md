# PILOT H2 tables (scores_pretrained)

PILOT (Amendment 2): partial data (VISION + 6 Synthbuster generators), not a Phase 4 verdict. alpha = 1%. Flag fake iff score > threshold. LODO/LOPO details in notes.md.

## clipdet  [FPR model: pretrained]

### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)

| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |
|---|---|---|---|---|---|
| M0 | 6.42% | 69/1074 | [5.03%, 8.06%] | 21.43% (D21) | [4.56%, 8.38%] |
| M1 | 2.51% | 27/1074 | [1.66%, 3.64%] | 16.67% (D21) | [1.49%, 3.63%] |
| M2 | 1.12% | 12/1074 | [0.58%, 1.94%] | 9.52% (D21) | [0.56%, 1.77%] |
| M3 | 0.09% | 1/1074 | [0.00%, 0.52%] | 2.38% (D12) | [0.00%, 0.28%] |
| M4 | 0.56% | 6/1074 | [0.21%, 1.21%] | 7.14% (D21) | [0.19%, 1.02%] |

### LOPO FPR (held-out platform; CP 95% CI)

| platform (n) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| native (358) | 10.06% [7.14%, 13.65%] | 4.19% [2.36%, 6.82%] | 2.51% [1.16%, 4.72%] | 1.12% [0.31%, 2.84%] | 1.68% [0.62%, 3.61%] |
| WA (358) | 3.63% [1.95%, 6.13%] | 1.40% [0.46%, 3.23%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| FBH (358) | 5.59% [3.45%, 8.50%] | 1.96% [0.79%, 3.99%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| jpeg75_resize05 (358) | 55.87% [50.55%, 61.08%] | 38.27% [33.21%, 43.52%] | 27.09% [22.56%, 32.02%] | 3.35% [1.74%, 5.78%] | 14.80% [11.29%, 18.91%] |
| screenshot_sim (358) | 22.35% [18.14%, 27.02%] | 9.78% [6.90%, 13.33%] | 6.42% [4.12%, 9.48%] | 0.84% [0.17%, 2.43%] | 2.79% [1.35%, 5.08%] |

Thresholds on all 25 test devices (clean reals; M1 from the 10 calib devices): M0=0.000, M1=0.420, M2=0.726, M3=1.411, M4=0.959

### TPR (pooled over all 720 fakes (pretrained detector)); thresholds fit on all 25 test devices' clean reals

| condition | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| clean (pooled) | 44.6% | 27.6% | 16.4% | 3.9% | 9.9% |
| jpeg75_resize05 (pooled) | 56.2% | 37.1% | 26.0% | 5.7% | 16.2% |
| screenshot_sim (pooled) | 85.1% | 76.1% | 62.4% | 37.1% | 53.8% |

TPR ratio M3/M2: clean 0.237, jpeg75_resize05 0.219

Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):

| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |
|---|---|---|---|---|---|---|
| dalle2 | 18.3% | 4.2% | 0.23 | 24.2% | 5.8% | 0.24 |
| dalle3 | 25.0% | 1.7% | 0.07 | 43.3% | 10.0% | 0.23 |
| firefly | 12.5% | 1.7% | 0.13 | 5.0% | 0.0% | 0.00 |
| midjourney-v5 | 12.5% | 3.3% | 0.27 | 21.7% | 5.0% | 0.23 |
| stable-diffusion-2 | 20.8% | 11.7% | 0.56 | 22.5% | 5.8% | 0.26 |
| stable-diffusion-xl | 9.2% | 0.8% | 0.09 | 39.2% | 7.5% | 0.19 |

### PILOT verdict: **FAIL**

- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: True
- M3 LOPO FPR <= 3% on every held-out platform: False
- TPR ratio >= 0.7 (clean 0.237, jpeg75_resize05 0.219): False
- some M0/M1/M2 FPR > 2% on a split: True (LODO/M0, LODO/M1, LOPO/native/M0, LOPO/native/M1, LOPO/native/M2, LOPO/WA/M0, LOPO/FBH/M0, LOPO/jpeg75_resize05/M0, LOPO/jpeg75_resize05/M1, LOPO/jpeg75_resize05/M2, LOPO/screenshot_sim/M0, LOPO/screenshot_sim/M1, LOPO/screenshot_sim/M2)
- fail triggers: M3 LODO > 2%: False; ratio < 0.5: True

## communityforensics  [FPR model: pretrained]

### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)

| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |
|---|---|---|---|---|---|
| M0 | 2.23% | 24/1074 | [1.44%, 3.31%] | 23.81% (D30) | [1.21%, 3.35%] |
| M1 | 1.49% | 16/1074 | [0.85%, 2.41%] | 19.05% (D30) | [0.74%, 2.33%] |
| M2 | 1.12% | 12/1074 | [0.58%, 1.94%] | 21.43% (D30) | [0.37%, 1.96%] |
| M3 | 0.37% | 4/1074 | [0.10%, 0.95%] | 9.52% (D30) | [0.09%, 0.74%] |
| M4 | 0.74% | 8/1074 | [0.32%, 1.46%] | 16.67% (D30) | [0.19%, 1.40%] |

### LOPO FPR (held-out platform; CP 95% CI)

| platform (n) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| native (358) | 4.19% [2.36%, 6.82%] | 2.23% [0.97%, 4.36%] | 1.96% [0.79%, 3.99%] | 1.68% [0.62%, 3.61%] | 1.68% [0.62%, 3.61%] |
| WA (358) | 1.12% [0.31%, 2.84%] | 1.12% [0.31%, 2.84%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| FBH (358) | 1.40% [0.46%, 3.23%] | 1.12% [0.31%, 2.84%] | 0.28% [0.01%, 1.55%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| jpeg75_resize05 (358) | 4.47% [2.58%, 7.16%] | 3.07% [1.54%, 5.43%] | 0.84% [0.17%, 2.43%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| screenshot_sim (358) | 5.03% [3.01%, 7.83%] | 3.91% [2.15%, 6.47%] | 2.23% [0.97%, 4.36%] | 0.56% [0.07%, 2.00%] | 0.84% [0.17%, 2.43%] |

Thresholds on all 25 test devices (clean reals; M1 from the 10 calib devices): M0=0.000, M1=1.069, M2=1.855, M3=7.588, M4=5.128

### TPR (pooled over all 720 fakes (pretrained detector)); thresholds fit on all 25 test devices' clean reals

| condition | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| clean (pooled) | 95.7% | 93.9% | 92.6% | 72.2% | 81.9% |
| jpeg75_resize05 (pooled) | 55.7% | 49.4% | 43.8% | 12.9% | 23.1% |
| screenshot_sim (pooled) | 96.8% | 95.7% | 94.3% | 72.9% | 85.0% |

TPR ratio M3/M2: clean 0.780, jpeg75_resize05 0.295

Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):

| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |
|---|---|---|---|---|---|---|
| dalle2 | 82.5% | 40.0% | 0.48 | 54.2% | 14.2% | 0.26 |
| dalle3 | 85.8% | 36.7% | 0.43 | 15.0% | 0.0% | 0.00 |
| firefly | 90.8% | 76.7% | 0.84 | 71.7% | 40.8% | 0.57 |
| midjourney-v5 | 98.3% | 91.7% | 0.93 | 22.5% | 5.0% | 0.22 |
| stable-diffusion-2 | 98.3% | 90.0% | 0.92 | 44.2% | 4.2% | 0.09 |
| stable-diffusion-xl | 100.0% | 98.3% | 0.98 | 55.0% | 13.3% | 0.24 |

### PILOT verdict: **FAIL**

- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: True
- M3 LOPO FPR <= 3% on every held-out platform: True
- TPR ratio >= 0.7 (clean 0.780, jpeg75_resize05 0.295): False
- some M0/M1/M2 FPR > 2% on a split: True (LODO/M0, LOPO/native/M0, LOPO/native/M1, LOPO/jpeg75_resize05/M0, LOPO/jpeg75_resize05/M1, LOPO/screenshot_sim/M0, LOPO/screenshot_sim/M1, LOPO/screenshot_sim/M2)
- fail triggers: M3 LODO > 2%: False; ratio < 0.5: True

## rine  [FPR model: pretrained]

### LODO FPR (pooled over 25 held-out devices; 1,074 clean reals)

| rule | FPR | k/n | CP 95% CI | max per-device FPR (device) | photo-cluster bootstrap 95% CI (suppl.) |
|---|---|---|---|---|---|
| M0 | 1.12% | 12/1074 | [0.58%, 1.94%] | 8.89% (D26) | [0.37%, 2.05%] |
| M1 | 0.84% | 9/1074 | [0.38%, 1.58%] | 8.89% (D26) | [0.28%, 1.58%] |
| M2 | 1.02% | 11/1074 | [0.51%, 1.83%] | 8.89% (D13) | [0.37%, 1.86%] |
| M3 | 0.09% | 1/1074 | [0.00%, 0.52%] | 2.22% (D26) | [0.00%, 0.28%] |
| M4 | 0.65% | 7/1074 | [0.26%, 1.34%] | 8.89% (D26) | [0.19%, 1.21%] |

### LOPO FPR (held-out platform; CP 95% CI)

| platform (n) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| native (358) | 2.23% [0.97%, 4.36%] | 1.96% [0.79%, 3.99%] | 3.35% [1.74%, 5.78%] | 0.84% [0.17%, 2.43%] | 1.96% [0.79%, 3.99%] |
| WA (358) | 0.56% [0.07%, 2.00%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| FBH (358) | 0.56% [0.07%, 2.00%] | 0.56% [0.07%, 2.00%] | 0.56% [0.07%, 2.00%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| jpeg75_resize05 (358) | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |
| screenshot_sim (358) | 0.56% [0.07%, 2.00%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] | 0.00% [0.00%, 1.03%] |

Thresholds on all 25 test devices (clean reals; M1 from the 10 calib devices): M0=0.000, M1=0.568, M2=0.568, M3=1.258, M4=0.729

### TPR (pooled over all 720 fakes (pretrained detector)); thresholds fit on all 25 test devices' clean reals

| condition | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| clean (pooled) | 64.4% | 61.5% | 61.5% | 53.6% | 58.6% |
| jpeg75_resize05 (pooled) | 48.2% | 40.7% | 40.7% | 21.2% | 34.7% |
| screenshot_sim (pooled) | 49.6% | 44.3% | 44.3% | 30.8% | 40.0% |

TPR ratio M3/M2: clean 0.871, jpeg75_resize05 0.522

Per-generator TPR (M2 -> M3; clean | jpeg75_resize05):

| generator | clean M2 | clean M3 | ratio | jpeg M2 | jpeg M3 | ratio |
|---|---|---|---|---|---|---|
| dalle2 | 63.3% | 52.5% | 0.83 | 42.5% | 22.5% | 0.53 |
| dalle3 | 0.8% | 0.0% | 0.00 | 32.5% | 17.5% | 0.54 |
| firefly | 41.7% | 26.7% | 0.64 | 40.0% | 21.7% | 0.54 |
| midjourney-v5 | 86.7% | 73.3% | 0.85 | 27.5% | 8.3% | 0.30 |
| stable-diffusion-2 | 78.3% | 71.7% | 0.91 | 40.0% | 18.3% | 0.46 |
| stable-diffusion-xl | 98.3% | 97.5% | 0.99 | 61.7% | 39.2% | 0.64 |

### PILOT verdict: **INCONCLUSIVE**

- M3 LODO FPR <= 1.5% and CP upper <= 2.0%: True
- M3 LOPO FPR <= 3% on every held-out platform: True
- TPR ratio >= 0.7 (clean 0.871, jpeg75_resize05 0.522): False
- some M0/M1/M2 FPR > 2% on a split: True (LOPO/native/M0, LOPO/native/M2)
- fail triggers: M3 LODO > 2%: False; ratio < 0.5: False
