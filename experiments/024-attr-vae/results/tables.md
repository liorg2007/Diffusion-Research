# H-A6 results (non-counting diagnostic)

**Label: INCONCLUSIVE**  inputs {"clean_bacc": 0.9674796747967479, "jpeg75_resize05_clean_src": 0.3333333333333333, "jpeg75_resize05_matched_src": 0.5839528716142777}

Gates: {"sha256_unique": true, "crop_hash_unique": true, "split_grouped": true, "permutation_at_chance": true, "m0_on_tensors_at_chance": true, "d008_ok": true}. sdxl features from `cpu`, sd-vae features (capped set) from `cpu`; capped=True; n=360. Chance bacc 0.333.

## 3-way bacc (95% cluster-bootstrap CI over caption families)

| condition | source | 8-feat LR | sd-vae only | sdxl only | AEDR-2 | e1-2 | M1 texture | diff vs M1 [CI] | M0 tensor | perm mean (p) | recalls kl-f8/XL/other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | clean | 0.967 [0.933, 0.993] | 0.748 [0.687, 0.809] | 0.718 [0.651, 0.791] | 0.950 [0.912, 0.985] | 0.967 [0.932, 0.993] | 0.754 [0.675, 0.828] | +0.214 [+0.134, +0.299] | 0.333 | 0.327 (0.005) | 1.00/1.00/0.90 |
| jpeg75_resize05 | clean | 0.333 [0.333, 0.333] | 0.333 [0.333, 0.333] | 0.396 [0.344, 0.451] | 0.317 [0.289, 0.333] | 0.333 [0.333, 0.333] | 0.422 [0.375, 0.473] | -0.088 [-0.140, -0.042] | 0.333 | 0.335 (0.557) | 0.00/0.00/1.00 |
| jpeg75_resize05 | matched | 0.584 [0.487, 0.672] | 0.552 [0.461, 0.642] | 0.564 [0.470, 0.654] | 0.508 [0.412, 0.600] | 0.460 [0.384, 0.544] | 0.735 [0.653, 0.822] | -0.151 [-0.250, -0.056] | 0.333 | 0.322 (0.005) | 0.68/0.64/0.44 |
| social_sim | clean | 0.333 [0.333, 0.333] | 0.379 [0.312, 0.451] | 0.390 [0.310, 0.468] | 0.317 [0.292, 0.333] | 0.325 [0.306, 0.333] | 0.515 [0.436, 0.594] | -0.181 [-0.260, -0.102] | 0.333 | 0.333 (0.478) | 0.00/0.00/1.00 |
| social_sim | matched | 0.591 [0.494, 0.678] | 0.489 [0.394, 0.587] | 0.492 [0.392, 0.585] | 0.445 [0.346, 0.549] | 0.473 [0.388, 0.562] | 0.701 [0.617, 0.785] | -0.110 [-0.218, -0.004] | 0.350 | 0.324 (0.005) | 0.62/0.67/0.49 |

## Confusion matrices (rows = true kl-f8, XL, other; cols = predicted)

- clean|src=clean: [[34, 0, 0], [0, 33, 0], [3, 1, 37]]  AUROC one-vs-rest [0.976, 1.0, 0.972]
- jpeg75_resize05|src=clean: [[0, 0, 34], [0, 0, 33], [0, 0, 41]]  AUROC one-vs-rest [0.541, 0.395, 0.344]
- jpeg75_resize05|src=matched: [[23, 7, 4], [9, 21, 3], [8, 15, 18]]  AUROC one-vs-rest [0.812, 0.739, 0.722]
- social_sim|src=clean: [[0, 0, 34], [0, 0, 33], [0, 0, 41]]  AUROC one-vs-rest [0.661, 0.465, 0.402]
- social_sim|src=matched: [[21, 7, 6], [4, 22, 7], [9, 12, 20]]  AUROC one-vs-rest [0.807, 0.745, 0.65]

## Argmin rules (no training)

| condition | A2 SD2-vs-SDXL bacc (chance 0.5) | A3 3-way bacc | tau | A3 recalls |
|---|---|---|---|---|
| clean | 1.000 [1.000, 1.000] | 0.740 [0.668, 0.813] | 0.391 | 0.94/0.42/0.85 |
| jpeg75_resize05 | 0.453 [0.366, 0.531] | 0.206 [0.134, 0.282] | 0.855 | 0.06/0.24/0.32 |
| social_sim | 0.484 [0.412, 0.561] | 0.303 [0.233, 0.379] | 0.811 | 0.03/0.52/0.37 |

## Split-seed sensitivity (8-feat LR, point estimates over seeds 1..10)

- clean|src=clean: mean 0.950 sd 0.024 range [0.900, 0.981]
- jpeg75_resize05|src=clean: mean 0.333 sd 0.000 range [0.333, 0.333]
- jpeg75_resize05|src=matched: mean 0.610 sd 0.039 range [0.565, 0.680]
- social_sim|src=clean: mean 0.337 sd 0.004 range [0.333, 0.343]
- social_sim|src=matched: mean 0.567 sd 0.045 range [0.499, 0.624]

## Secondary: SD1.3 / SD1.4 (not in training); share predicted per family [CI over caption families]

| condition | source | generator | n | kl-f8 | XL | other |
|---|---|---|---|---|---|---|
| clean | clean | stable-diffusion-1-3 | 50 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| clean | clean | stable-diffusion-1-4 | 50 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| jpeg75_resize05 | clean | stable-diffusion-1-3 | 50 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| jpeg75_resize05 | clean | stable-diffusion-1-4 | 50 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| jpeg75_resize05 | matched | stable-diffusion-1-3 | 50 | 0.38 [0.26, 0.52] | 0.14 [0.06, 0.24] | 0.48 [0.34, 0.62] |
| jpeg75_resize05 | matched | stable-diffusion-1-4 | 50 | 0.28 [0.16, 0.42] | 0.08 [0.02, 0.16] | 0.64 [0.50, 0.78] |
| social_sim | clean | stable-diffusion-1-3 | 50 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| social_sim | clean | stable-diffusion-1-4 | 50 | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.06] | 0.98 [0.94, 1.00] |
| social_sim | matched | stable-diffusion-1-3 | 50 | 0.60 [0.48, 0.74] | 0.04 [0.00, 0.10] | 0.36 [0.22, 0.48] |
| social_sim | matched | stable-diffusion-1-4 | 50 | 0.46 [0.32, 0.60] | 0.04 [0.00, 0.10] | 0.50 [0.36, 0.64] |

Reference (main test set, same model): share predicted kl-f8 for true SD2 / SDXL / other

- clean|src=clean: SD2 1.00, SDXL 0.00, other 0.07 (n_test per class [34, 33, 41])
- jpeg75_resize05|src=clean: SD2 0.00, SDXL 0.00, other 0.00 (n_test per class [34, 33, 41])
- jpeg75_resize05|src=matched: SD2 0.68, SDXL 0.27, other 0.20 (n_test per class [34, 33, 41])
- social_sim|src=clean: SD2 0.00, SDXL 0.00, other 0.00 (n_test per class [34, 33, 41])
- social_sim|src=matched: SD2 0.62, SDXL 0.12, other 0.22 (n_test per class [34, 33, 41])

## sd-vae only, full 1,020-row set (capped run)

- clean|src=clean: sd-vae 4-feat bacc 0.775 [0.722, 0.821]  (8-col model with the sdxl columns zeroed = sd-vae only)
- jpeg75_resize05|src=clean: sd-vae 4-feat bacc 0.329 [0.323, 0.333]  (8-col model with the sdxl columns zeroed = sd-vae only)
- jpeg75_resize05|src=matched: sd-vae 4-feat bacc 0.612 [0.529, 0.688]  (8-col model with the sdxl columns zeroed = sd-vae only)
- social_sim|src=clean: sd-vae 4-feat bacc 0.304 [0.256, 0.359]  (8-col model with the sdxl columns zeroed = sd-vae only)
- social_sim|src=matched: sd-vae 4-feat bacc 0.501 [0.414, 0.578]  (8-col model with the sdxl columns zeroed = sd-vae only)

## D-008
```
{
 "pixel": {
  "sd-vae-ft-mse": {
   "n": 32,
   "max_abs_dpixel_x1": 0.00012305378913879395,
   "max_abs_dpixel_x2": 5.874037742614746e-05,
   "tol": 0.001,
   "ok": true
  }
 },
 "feature_level": {
  "all_accelerator_vs_cpu": {
   "clean": {
    "max_abs_dlogit_both": 0.15963691873158836,
    "max_abs_dfeature": 0.011520220040788565,
    "max_abs_dfeature_sdvae_cols": 0.0005483549475466809,
    "max_abs_dfeature_sdxl_cols": 0.011520220040788565,
    "max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu": 0.0016382735675595228
   },
   "jpeg75_resize05": {
    "max_abs_dlogit_both": 0.14926516747534735,
    "max_abs_dfeature": 0.010938823562197442,
    "max_abs_dfeature_sdvae_cols": 0.00038822718511855037,
    "max_abs_dfeature_sdxl_cols": 0.010938823562197442,
    "max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu": 0.004324882852571932
   },
   "social_sim": {
    "max_abs_dlogit_both": 0.2261795445790431,
    "max_abs_dfeature": 0.01669746037214015,
    "max_abs_dfeature_sdvae_cols": 0.0002348823339639594,
    "max_abs_dfeature_sdxl_cols": 0.01669746037214015,
    "max_abs_dlogit_sdvae_gpu_sdxl_cpu_vs_all_cpu": 0.001459740436121848
   }
  },
  "ok_all_accelerator": false,
  "ok_sdvae_accelerator_sdxl_cpu": false
 },
 "note_all_cpu": "capped analysis set: both VAEs on CPU -> no accelerator features in the model; the accelerator-vs-CPU feature guard failed for sdxl (0.15 logit) and marginally for sd-vae (1.1e-3..3.0e-3 vs 1e-3), which is why the sd-vae features were recomputed on CPU for the subset"
}
```
