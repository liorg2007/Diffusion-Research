# A Bias-Free Training Paradigm for More General AI-generated Image Detection (B-Free)
- **id:** bfree-cvpr2025
- **Authors / venue / year:** Fabrizio Guillaro, Giada Zingarini, Ben Usman, Avneesh Sud, Davide Cozzolino, Luisa Verdoliva (Univ. Federico II Naples; Google DeepMind). CVPR 2025 (venue stated in the official GitHub README; the arXiv HTML v2 page itself does not print a venue).
- **Links:** arXiv 2412.17671 (v2, 3 Apr 2025) · https://arxiv.org/abs/2412.17671 · code https://github.com/grip-unina/B-Free
- **Family:** data-bias-benchmarks
- **Status:** partially-verified (arXiv HTML v2 read through a fetch/summarisation tool; the PDF could not be text-extracted locally, so table numbers below were not re-checked against PDF)

## Summary (plain language)
Detectors often learn shortcuts in the training data (image content, file format, resolution) instead of generator traces. B-Free builds fake images *from* real ones (Stable Diffusion 2.1 inpainting that regenerates the same image, optionally swapping objects), so real and fake match in content, format and size, and adds heavy augmentation. A DINOv2 ViT trained this way is reported to generalise to many unseen generators and to social-network images, with better calibration than competitors.

## How it works
Real image -> SD 2.1 inpainting with an empty mask ("self-conditioning", regenerates the whole image) or an object/rectangle mask ("content augmentation"; optionally restore the original background). Classifier: ViT with 4 registers, DINOv2 pretraining, trained end-to-end (not frozen) on 504x504 crops without resizing; at test time 504x504 crops, multi-crop averaging for larger images (GitHub README). Final recipe "inpainted++" = self-conditioned + local inpainting + blur, JPEG, scaling, cut-out, noise, jitter (README).

## Training data
51,517 real MS-COCO train images (Creative-Commons-licensed only; largest central crop resized to 512x512) and 309,102 SD 2.1-inpainting fakes (six fake versions per real image per arXiv HTML; the 309,102 / 51,517 = 6.0 ratio is consistent). Watermark not embedded. Blur/JPEG/scaling/etc. are applied at training time, not stored in the released set (GitHub README, "Training set").

## Reported results
- Table 5 (balanced accuracy bAcc / balanced NLL per generator): B-Free averages 96.3% bAcc, NLL 0.10; paper states a 20.7-point bAcc gain over the second-best method. WildRF bAcc: Facebook 95.6, Reddit 86.2, Twitter 97.3 (Table 5).
- Table 7 (GenImage-unbiased, FakeInversion, SynthWildX averages): 90.4% bAcc, NLL 0.24.
- Table 2 ablation (Synthbuster + WildRF): best variant (inpainted++) 96.4% average bAcc.
- Metrics: AUC and accuracy at a FIXED 0.5 threshold, plus ECE (15 bins) and balanced NLL - i.e. this paper does report threshold-fixed, calibration-aware numbers, but on fixed test sets, not on a fresh real-photo collection.
- **NPR claim verification (Table 11).** Table 11 is captioned "Performance on FakeInversion dataset" and compares *Original* vs *Social network simulation*. For NPR: Original AUC/bAcc = 91.6 / 87.0 (NLL/ECE 4.96 / .123); Social-network simulation AUC/bAcc = **43.3 / 49.9** (NLL/ECE 27.1 / .501). So the "~49.9%" is real but it is **balanced accuracy on FakeInversion only** (AUC 43.3, below chance), not a general or multi-dataset number. The simulation, quoted from the paper: "we resize with a scale factor randomly sampled between 0.7 and 1, and compress with a JPEG quality factor between 70 and 100" - a *mild* resize (0.7-1.0) plus JPEG Q70-100, applied once. The appendix text says performance "drops substantially, except for DMID and our method" without naming NPR. Table 11 is in the appendix (PDF p.13; the simulation sentence is on PDF p.12; the main-text reference to it was not located). Note 49.9 also appears as NPR's SynthWildX bAcc in Table 7 (p.8), a different cell.
- Unseen-generator note: NPR row in Table 5 (PDF p.7) has bAcc 0.6 on DALL-E 3 and 0.4 on Firefly; NPR Table 5 average 42.7 bAcc. UnivFD/FatFormer/RINE Table 5 averages 62.4/52.9/65.0; Table 7 averages (GenImage-unbiased, FakeInversion, SynthWildX) NPR 62.6, UnivFD 56.9, FatFormer 58.2, RINE 63.0.

## Known weaknesses / failure modes
- Self-reported by authors: not fully specified in the portions read; Reddit (86.2 bAcc) is the weakest WildRF platform (Table 5).
- B-Free's training fakes come from ONE generator family (SD 2.1 inpainting), so unseen-generator claims rest on the bias-free design + augmentation.
- Later independent evaluations: none located in this pass **[unverified]**.

## Code + weights
Code: https://github.com/grip-unina/B-Free . Licence (LICENSE.txt read): GRIP-UNINA custom licence - use, reproduction, modification allowed free of charge only for "informational and nonprofit purposes"; industrial/profit use prohibited -> **NON-COMMERCIAL**; GitHub API reports spdx NOASSERTION. Weights: code/README.md points to https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip (MD5 f3f53fa647848b16cf81c913f148a198; contains config.yaml and model_epoch_best.pth). Publicly downloadable (link not downloaded by me; size unknown). Released data: training set, viral-images set (~1,400 images), extended Synthbuster (RAISE-1k resized + FLUX + SD3.5) - see kb/datasets/bfree-extended-synthbuster.md.

## Verified?
Partially. Table 11 values and the quoted simulation sentence were returned from the arXiv HTML v2 page; licence read from LICENSE.txt in the repo; weights path read from code/README.md. Table 5/7/2 numbers come from the same fetch (not PDF-checked).
(verified vs PDF p.13 and p.12, 2026-10-21) Table 11 NPR row: Original AUC/bAcc 91.6/87.0, NLL/ECE 4.96/.123; social-network simulation 43.3/49.9, NLL/ECE 27.1/.501; simulation sentence (resize 0.7-1, JPEG 70-100) exact.
(verified vs PDF p.7-8, 2026-10-21) Table 5: B-Free WildRF 95.6/86.2/97.3, avg 96.3 bAcc / 0.10 NLL, +20.7 over second best; Table 7: B-Free avg 90.4 / 0.24; NPR/UnivFD/FatFormer/RINE rows read directly (see kb/verification-log.md). Table 2 (96.4) not re-checked.

## Relevance to this project
- Key reference for "bias-free" training and for evaluating with reals that match fakes in format/size; our eval set should be format-matched.
- The NPR collapse under a mild resize+JPEG simulation (bAcc 49.9, FakeInversion) is a concrete motivation for testing robustness, but cite it only as FakeInversion/one-setting evidence.
- Its released extended-Synthbuster (RAISE + 11 generators incl. FLUX, SD3.5) is a ready ~4.7 GB eval source.
