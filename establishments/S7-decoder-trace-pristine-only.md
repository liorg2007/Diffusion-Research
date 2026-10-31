# S7: decoder traces exist only on pristine files

Running an image through candidate VAEs identifies its decoder on untouched files: 3-way **0.967** [0.933, 0.993], and a training-free rule separates SD2 vs SDXL at 1.000. After one JPEG-75 + half-size resize, clean-trained accuracy is chance (0.333) and the training-free rule falls to 0.45.

It identifies specific decoder weights, not architecture families (Midjourney v5 matches sd-vae-ft-mse; SD1.x does not). This is a non-counting diagnostic with an INCONCLUSIVE label.

The noise-residual fingerprint for generators behaves similarly: 0.522 clean, tied with texture statistics, and locked to the image corner.

**Evidence.** `experiments/024-attr-vae/results/{tables.md,critique.md}`, `experiments/021-attr-residual/results/critique.md`. Consistent with DNA (arXiv 2026), reported at 9.15% under JPEG-75.
