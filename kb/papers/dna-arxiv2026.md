# DNA: Dual-stage Native Attribution for Generated Image Source Tracing
- **id:** dna-arxiv2026
- **Authors / venue / year:** Chao Wang, Kejiang Chen, Zijin Yang, Yaofei Wang, Yuang Qi, Weiming Zhang, Nenghai Yu / arXiv preprint (journal extension of AEDR, no journal named) / 2026
- **Links:** arXiv 2607.13685v1 (15 Jul 2026) · code https://github.com/wangchao0708/DNA
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Two-stage training-free attribution. Stage 1 (AEDR) finds the model FAMILY from VAE double-reconstruction; Stage 2 (NPC) tells variants in the family apart by comparing the backbone's native prediction errors over several noise timesteps. Includes the first real-world JPEG test of this line, and it is bad news for Stage 1.

## How it works
Stage 1: one VAE per family, open-set screening. Stage 2: forward-only access to each candidate backbone in the predicted family (no gradients); BLIP-2 captions supply semantic conditions; default calibration n=100 per variant (500 per model in DNA-30K; unreconciled in text).

## Training data
None. Benchmark DNA-30K: 24 candidate models in 6 families (list not read).

## Reported results
- End-to-end 89.11% avg (Tab. VI; best baseline De-Fake-Hyb 6.77%, random 0.40%). Stage 1 joint decision 92.25% (Tab. V; LatentTracer 23.85%); per-family binary AUC 99.54% (Tab. IV); Stage 2 oracle 96.36% (Sec. VII-B2).
- Robustness (Sec. VII-E): real-world JPEG QF=75 on SD2.x: end-to-end 9.15%; DNA-DCT variant 34.60%. Facebook pilot, 100 images: DNA-DCT 3% -> 31%. Resize: not reported. Scheduler changes: SDXL 84.25-87.90%.
- Cost (Tab. IX): Stage 1 ~0.107 s/image; Stage 2 1.031 s (SD2.x) to 86.68 s (FLUX.2). Hardware not read.
- Unknown-variant rejection: AUROC 0.651 (Sec. VIII).

## Known weaknesses / failure modes
Stage 1 pixel-level signals are "more sensitive to compression artifacts" (authors); need candidate weights for all variants; closed-set within families.

## Code + weights
Repo above; licence not stated in text read **[unverified]**.

## Verified?
Partially: fetch-tool summary of arXiv HTML (first 100k chars plus second chunk).

## Relevance to this project
- Best evidence that VAE-reconstruction attribution collapses under JPEG75 (9.15% end-to-end), the condition we require.
- Stage 2 idea (backbone prediction error) needs full diffusion U-Nets: out of reach for our two VAEs.
