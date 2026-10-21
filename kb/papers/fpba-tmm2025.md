# Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks (FPBA)
- **id:** fpba-tmm2025
- **Authors / venue / year:** Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang. arXiv page says "Accepted in TMM" (IEEE Trans. Multimedia; not independently confirmed); arXiv v1 Jul 2024, v6 Feb 2026.
- **Links:** arXiv 2407.20836 · https://arxiv.org/abs/2407.20836 · code: abstract says public on GitHub; URL not found in fetched text [unverified]
- **Family:** robustness-attacks
- **Status:** partially-verified

## Summary (plain language)
Proposes FPBA, a transfer attack that perturbs both frequency (DCT) and spatial domains and uses a post-trained Bayesian ensemble of a single surrogate to imitate unknown victims. Evaluated on 13 detectors, including cross-generator and JPEG-compressed settings.

## How it works
L-inf eps=8/255, 10 iterations, step 2/255. White-box and black-box transfer. Frozen surrogate plus appended Bayesian MLP components (K=3), random spectrum transforms (N=5).

## Training data
Detectors trained by the authors/standard recipes on: ProGAN LSUN (360k real + 360k fake); GenImage subset (ImageNet reals + SD v1.4 fakes; tested on Midjourney, SD1.4/1.5, ADM, Wukong, BigGAN); FFHQ vs StyleGAN2 (50k+50k). Detectors: CNNSpot, MobileNet, EfficientNet, DenseNet, ViT, Swin, DCTA, Spec, LGrad, UnivFD, DNF, LNP, GramNet.

## Reported results
- Table I: avg white-box success 99.6% (FPBA) vs 93.1% (S2I, PGD); avg black-box 50.5%, +8.4 over PGD.
- Table VIII: under Blur+JPEG training at 50% prob, FPBA 99.2% success vs 86.6% IFGSM.
- Test-time JPEG QF 30-100 step 10 (Fig 6) [numbers not extracted].
- Table X: PSNR 36.26 dB, SSIM 0.94 (PGD 33.51 / 0.88).
- Table VI: cross-generator transfer from Swin-ViT averages 28.0% on CNNSpot, 12.7% on ViT.
- Adversarial training attempted, reported not to converge (so no real adversarial-training baseline).

## Known weaknesses / failure modes
Black-box success only ~50% on average; much lower across architectures/generators. Cross-generator transfer is weak (12-28%).

## Code + weights
Not verified.

## Verified?
Partially: arXiv HTML v3 via fetch summary; the page text said "code will be shared upon acceptance" while the abstract says it is public.

## Relevance to this project
- Blur/JPEG augmentation (10-50%) does not defend against adversarial attack.
- Defense-by-adversarial-training for detectors is under-explored; useful negative evidence.
