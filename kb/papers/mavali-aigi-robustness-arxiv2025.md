# Adversarial Robustness of AI-Generated Image Detectors in the Real World (v1 title: "Fake It Until You Break It: On the Adversarial Robustness of AI-generated Image Detectors")
- **id:** mavali-aigi-robustness-arxiv2025
- **Authors / venue / year:** Sina Mavali, Jonas Ricker, David Pape, Asja Fischer, Lea Schonherr (CISPA / Ruhr U. Bochum). arXiv preprint, v1 Oct 2024, v3 3 Jun 2025. No venue stated.
- **Links:** arXiv 2410.01574 · https://arxiv.org/abs/2410.01574 · code: paper links github.com/smavali/AIGI-Break; search result also pointed to github.com/dormant-neurons/aigi-break; neither resolved via GitHub API (404) [unverified]
- **Family:** robustness-attacks
- **Status:** partially-verified

## Summary (plain language)
Evaluates five black-box-style attacks (mostly transfer) against four detectors, including with social-media-like degradation after the attack, and on HIVE, a commercial detector. Also tests a defense: swapping UnivFD's CLIP for an adversarially fine-tuned CLIP.

## How it works
Threat model: black-box (transfer from surrogate ensemble; one query-based attack). Attacks (L-inf, eps=8/255 main; 4 and 16 in quality analysis): PGD (10 steps, step 2/255); Ensemble with Leave-Target-Out; Diverse Input (resize/padding p=0.8, scale 0.8-1.2); Universal perturbation (50 steps); query-efficient NES (max 500 queries, 50 samples/grad, sigma=0.005). Degradations after attack, following Xu et al.: Medium = downsample to 75% + JPEG Q50; High = 5x5 Gaussian blur (sigma=1) + downsample to 50% + JPEG Q20 + Gaussian noise (sigma=0.1). Detectors: DRCT-CLIP, DRCT-ConvB, UnivFD, Corvi.

## Training data
Detectors used as released. Eval data: Synthbuster (1,000 imgs from nine diffusion generators, RAISE-1k reals), Chameleon, and GPT-4o images (generated 3 Apr 2025, reuse Synthbuster reals).

## Reported results
- Benign mean ROC AUC (Table I): DRCT-CLIP 92.3, DRCT-ConvB 83.8, UnivFD 81.5, Corvi 78.1.
- SD-1.4, eps 8/255, no degradation (Table II) average ROC AUC: PGD 58.6; Ensemble(LTO) 23.4 (Corvi stays 84.6); Diverse Input 4.1 (Corvi 14.4); Universal 46.9; Query-efficient 79.7.
- With degradation: Diverse Input more robust than Ensemble; Universal least robust (per-number table not extracted: unverified).
- HIVE (Sec IV-E, Table V; Diverse Input, 20% of images): ROC AUC 99.0 -> 73.8 (Synthbuster), 98.2 -> 79.7 (Chameleon); real-image accuracy 99% -> 52%, fake 96% -> 84%.
- Defense (Table VI): UnivFD with RobustCLIP (R2: fine-tuned eps 2/255; R4: 4/255). Under Diverse Input ROC AUC ~0 -> 69.0 (R2), 75.7 (R4); benign AUC 91.4 -> 88.4 (R2), 84.4 (R4). Averages under attack + high degradation: 55.7 / 73.8 / 74.6.
- Last ~1,200 chars of the HTML page were not read.

## Known weaknesses / failure modes
Defense costs 3-7 AUC points on benign data and is only tested against attack types studied. Attack mostly survives JPEG Q50 / 75% resize.

## Code + weights
Repo license not verified [unverified].

## Verified?
Partially: HTML v3 via fetch summary, not PDF.

## Relevance to this project
- Gives concrete, reproducible degradation recipes (Medium: 75%+Q50; High: blur+50%+Q20+noise) and eps=8/255 transfer settings.
- Shows robust-CLIP backbone as a cheap defense baseline; attacks affect real-image FPR too (HIVE reals 99%->52%).
