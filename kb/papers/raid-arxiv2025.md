# RAID: A Dataset for Testing the Adversarial Robustness of AI-Generated Image Detectors
- **id:** raid-arxiv2025
- **Authors / venue / year:** Eddoubi, Ricker, Cocchi, Baraldi, Sotgiu, Pintor, Cornia, Fischer, Cucchiara, Biggio (11 authors per arXiv metadata; full list not re-verified). arXiv preprint, June 2025. No peer-reviewed venue found.
- **Links:** arXiv 2506.03988 · https://arxiv.org/abs/2506.03988 · code https://github.com/pralab/RAID · data https://huggingface.co/datasets/aimagelab/RAID
- **Family:** robustness-attacks
- **Status:** partially-verified (abstract page + arXiv HTML read via fetch tool summary; PDF tables not read directly)

## Summary (plain language)
The authors attack an ensemble of seven AI-image detectors with PGD and release the resulting ~72k adversarial images as a ready-made benchmark. Because these images were crafted on other detectors, they transfer to unseen detectors and give a cheap estimate of a new detector's adversarial robustness without running attacks. Main claim: current detectors are easily fooled, including by transfer.

## How it works
Threat models: white-box (attacker knows target) and transfer/black-box (perturbation crafted on other detectors, tested on held-out one; "N-model" ensemble excludes the evaluated detector) (Sec 3.2, Tables 2-3). Attack: L-inf PGD, 10 steps, step size 0.05, ensemble loss = mean cross-entropy toward target class (Sec 3.1, Eq 1-3). Epsilon: released dataset 8/255; experiments report 16/255 and 32/255 (Sec 2.1, 3.1) [exact mapping of released-vs-experimental eps should be re-checked in PDF: unverified]. Attacked detectors: Wang20, Corvi23, Ojha23, Cavia24, Koutlis24, DRCT (CLIP and ConvNeXt backbones) (Sec 2.2). Validation baselines: DINOv2, DINOv2-Reg, ViT-Tiny, CoDE (Table 4).

## Training data
Not a detector; dataset built on D3. Generators: SD v1.4, SD v2.1, SDXL, DeepFloyd IF. 4,800 new real images -> 24,000 images per eps, 72,000 adversarial, 96,000 with originals (Sec 2.1).

## Reported results
- White-box eps=16/255 (Table 2): F1 -> 0.0 for all detectors except DRCT-ConvNeXt (Ch24CN, F1 ~0.29). eps=32/255 (Table 3): Ch24CN F1 ~0.25 vs 0.86 clean (Table 1).
- Transfer from K24 (Table 2): AUROC Ca24 0.84->0.56; Ch24C 0.73->0.51; Ch24CN 0.94->0.59; O23 0.95->0.50.
- N-model ensemble (Table 2): Co23 F1 ~0.98-0.99 -> 0.53 (AUROC 0.67); W20 -> 0.40 (AUROC 0.62). eps=32/255 (Table 3): AUROC 0.59 (Co23), 0.54 (W20).
- RAID vs clean on held-out baselines, eps=16/255 (Table 4): accuracy DINOv2 83.1->54.6; ViT-T 83.7->68.2; CoDE 91.9->75.6. PNG vs JPEG saving has little effect on attack success (RAID vs RAID_IMG).
- Commercial, preliminary, 50 pairs (Table 7): Sightengine accuracy on AI images 0.66->0.00; HIVE 0.60->0.44.
- No FPR-at-fixed-threshold on fresh real photos; no social-network laundering study.

## Known weaknesses / failure modes
Only an aggregate of attacks on existing detectors; DRCT-ConvNeXt was most resistant (possibly because of its own training augmentation) [interpretation unverified]. Perturbation budgets 16-32/255 are large vs. typical 4-8/255. Dataset limited to 4 generators (all SD-family/IF) and D3 reals.

## Code + weights
Repo MIT (GitHub API license check). HF dataset tagged license:mit, not gated (HF API). Only evaluation code; no detector weights of its own.

## Verified?
Partially: numbers from a fetch-tool summary of arXiv HTML (v3); not cross-checked against PDF tables. Licenses read from GitHub/HF APIs.

## Relevance to this project
- Gives a drop-in transfer-attack test set; use it to compare our detector's AUROC under transfer at eps 8/16/255.
- Shows high-accuracy clean detectors (AUROC 0.95) can fall to ~0.5 under transfer: clean unseen-generator numbers say nothing about adversarial robustness.
