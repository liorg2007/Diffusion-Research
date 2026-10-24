# Fragile Reconstruction: Adversarial Vulnerability of Reconstruction-Based Detectors for Diffusion-Generated Images
- **id:** fragile-recon-arxiv2026
- **Authors / venue / year:** Haoyang Jiang, Mingyang Yi, Shaolei Zhang, Junxian Cai, Qingbin Liu, Xi Chen, Ju Fan (Renmin Univ. of China, Tencent) / arXiv preprint (no venue) / 2026
- **Links:** arXiv 2604.12781 (v1) · https://arxiv.org/abs/2604.12781 · code https://github.com/atrijhy/Fragile-Reconstruction
- **Family:** robustness-attacks
- **Status:** partially-verified

## Summary (plain language)
White-box adversarial attacks (eps 8/255) drive DIRE, LaRE^2 and AEROBLADE to near-zero accuracy, and attacks transfer weakly or not at all between them in the black-box tables. The authors attribute it to low signal-to-noise in reconstruction features. Important for any reconstruction/VAE-trace detector we propose, since RAID-style transfer attacks are a Phase-5 target.

## How it works
APGD, 100 steps, l_inf eps = 8/255; inversion treated as a Neural ODE with adjoint gradients (Sec. 4.2-4.3). Detectors: DIRE, LaRE^2, AEROBLADE. Generators: ADM, SDv1.5, FLUX, VQDM (FLUX described as rectified flow). 5,000 real + 5,000 fake per generator for attack evaluation; 80,000 per generator for training.
- **Weights needed:** attacker needs the detector (white-box); n/a for us.
- **Passes per image:** n/a (attack paper).

## Training data
Detector training 80,000 images per generator; 10,000 held-out evaluation (App. 0.A.2).

## Reported results
- White-box (Fig. 3, no table): robust accuracy near 0% for all detectors on all four generators.
- Black-box transfer (Table 1, accuracy): DIRE->LaRE^2 50.39, DIRE->AEROBLADE 50.00, LaRE^2->DIRE 52.04, LaRE^2->AEROBLADE 50.00, AEROBLADE->DIRE 49.27, AEROBLADE->LaRE^2 49.93 (these are near chance; read as "transfer between these detectors is ineffective in the reported setup" - interpretation **[unverified]**).
- Adversarial training (Table 2, clean / robust %): DIRE ADM 64.4/0.0, FLUX 85.6/0.0; LaRE^2 ADM 74.0/23.1, FLUX 98.1/90.3. AEROBLADE has partial (~20%) robustness only at eps = 1/255 (App. 0.A.3).
- Diffusion purification defense left accuracy near 50% and lowered clean accuracy (DIRE 62.8, LaRE^2 51.1, AEROBLADE 53.8, Fig. 7 / text).
- **Unseen-generator results:** n/a. **Laundering results:** JPEG/resize purification NOT tested. **FPR on real photos:** not reported.

## Known weaknesses / failure modes
Only three detectors and four generators; conclusions for other VAE-only methods are inferred. Ambiguity about the Table 1 transfer direction (near-50% on both sides).

## Code + weights
- **License:** not stated in fetched text **[unverified]**.

## Verified?
Partially: arXiv HTML v1 (first 100k of 101.4k chars); Tables 1-2.

## Relevance to this project
- Reconstruction-error detectors (incl. AEROBLADE) have no adversarial robustness: our "robustness to transfer attacks" target needs an explicit plan (e.g. randomised or ensembled VAEs), not a free property.
- Includes FLUX as a tested generator for reconstruction detectors (LaRE^2 98.1 clean on their FLUX set, Table 2).
