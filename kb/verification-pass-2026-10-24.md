# Verification pass 2 (headline numbers, primary-source text, 2026-10-24)

Method: PDFs fetched from arxiv.org/pdf/<id> to a scratch directory outside the repo (not saved in the repo); text extracted with Ghostscript txtwrite; table cells read directly. Complements `kb/verification-log.md` (pass 1, 2026-10-21). Items 8 and 11 were already confirmed in pass 1 and were re-checked here only by a grep of the relevant cells.

| # | Claim | Source (URL, table/section) | Value found | Status |
|---|-------|-----------------------------|-------------|--------|
| 1 | NPR self-reported 92.2 mean Acc; 28 vs 38 sub-testsets | https://arxiv.org/abs/2312.10461 (v2), Table 6; Sec. 4.1/4.2 text | Table 6 header reads "Mean Acc. of 38 sub-testsets", NPR(our) = 92.2 (Ojha 79.8, LGrad 80.5). Text: "mean accuracy across 28 generation models, our NPR achieves 92.2% shown in Table 6, outperforming Ojha and LGrad by 12.4% and 11.7%". Eval set described as "28 generation models" in five datasets. | CONFIRMED (discrepancy is real: caption/table says 38 sub-testsets, text says 28 models) |
| 2 | SPAI Table 1 avg AUC | https://arxiv.org/abs/2411.19417, Table 1 (AVG column; AUC averaged over 5 real sources, 13 generators) | SPAI 91.0; NPR 45.4; RINE 85.5; DMID 83.5; UnivFD 67.3; LGrad 65.9; CNNDet. 55.5 | CONFIRMED |
| 3a | B-Free Table 11 AUC original -> social-sim | https://arxiv.org/abs/2412.17671, Table 11 (FakeInversion; cells AUC / bAcc) | NPR 91.6 -> 43.3 (bAcc 87.0 -> 49.9); FatFormer 68.2 -> 48.7; UnivFD 54.9 -> 49.7; RINE 69.6 -> 62.4; C2P-CLIP 74.1 -> 71.9 | CONFIRMED |
| 3b | B-Free Table 5 avg bAcc 96.3 | same, Table 5, AVG column | B-Free (ours) 96.3 bAcc / 0.10 NLL | CONFIRMED |
| 4a | Corvi Table 1 Grag2021 resize+JPEG | https://arxiv.org/abs/2211.00680, Table 1 | Pass 1 log: Acc 61.5 / AUC 90.8 (not re-read in this pass) | CONFIRMED (pass 1) |
| 4b | Corvi Table 2 Platt fusion 87.8/94.1 | same, Table 2, AVG row, "Calibration" column (fusion of Grag2021 LDM-trained + ProGAN-trained, then Platt scaling) | Acc./AUC = 87.8 / 94.1. Fusion alone (no calibration) is 64.5 / 94.1; LDM-only is 66.9 / 89.2. Column format is Acc./AUC. | CONFIRMED |
| 5a | Fake-or-JPEG Table 4 | https://arxiv.org/abs/2403.17608, Table 4, "Total" row | ResNet50 71.68 -> 82.74 (+11.06); Swin-T 74.09 -> 85.83 (+11.74) | CONFIRMED |
| 5b | Fake-or-JPEG Table 2 | same, Table 2 (ResNet50 trained on GenImage Midjourney subset, tested on 1024x1024 natural FFHQ) | Accuracy: PNG 80.45; JPEG95 94.84; JPEG90 98.93; JPEG80 99.95; JPEG70 99.99; JPEG60 100.0 | CONFIRMED |
| 6 | LAION-Mobile | https://arxiv.org/abs/2609.11134, Table 4, Sec. 5.2-5.3, Fig. 8, Sec. 3 | FPR at tau(NTIRE-EER) spans 0.174 (CNNDetection) to 0.913 (Effort) across the five detectors that were <=11% at tau(ProGAN-ISP): CNNDetection 17.4, UnivFD 39.1, CLIP-RB 42.9, IAPL 89.3, Effort 91.3. UnivFD 0.002 (0.2%) at ProGAN-ISP vs 0.391 (39.1%) at NTIRE. Effort 0.036 (3.6%) vs 0.913 (91.3%). 212 images (0.03%) from phones released 2021 or later. RIGID-DINOv3 NTIRE AUC 0.624 [0.612, 0.635] (best of 12). Text: "re-fitting ... lifts every one of them to 17-91%". | CONFIRMED (the "17.4%-91.3%" range covers those five detectors only; other detectors at tau(NTIRE) are 51.5-83.1%, and at tau(ProGAN-ISP) several are 94.6-97.9%) |
| 7 | Community Forensics Table 2 | https://arxiv.org/abs/2411.04125, Table 2, Sec. 4.2; Sec. 3 | "Ours - High res." on own Comprehensive set: mAP 0.987, Acc 0.892 (text: "0.987 mAP and 89.2% accuracy"). Eval set: 26K images from 21 models not in training set. Training set: 4803 models, 2.7M images. Low-res model: 0.971 / 0.861. | CONFIRMED (note: the 0.987 / 89.2% is the High-res (384) model; "4,803" appears as "4803" in the paper) |
| 8 | vfm-baseline Table 4 | https://arxiv.org/abs/2509.12995, Table 4 | DINOv3(2025.8.14): 97.5 real / 91.4 fake / 94.5 avg | CONFIRMED |
| 9a | RAID Table 2 (16/255) UnivFD (O23) | https://arxiv.org/abs/2506.03988, Table 2 (F1/AUROC; rows = attacked model, columns = evaluated model) | Attack on K24, evaluated on O23: 0.41 / 0.62. Ensemble attack (N-model row, O23 column): 0.18 / 0.54. White-box O23: 0.0 / 0.5. Clean 0.95 from pass 1 (Table 1). | CONFIRMED |
| 9b | RAID Table 4 DINOv2 | same, Table 4 (Accuracy) | D3 83.1 -> RAID(eps=16/255) 54.6; eps=32/255 43.6 | CONFIRMED |
| 10 | Manifold-bias ICLR 2025 Table 1 | https://arxiv.org/abs/2504.15470 (ICLR 2025 header), Table 1 | AUC: RIGID 0.439, AEROBLADE 0.444, Ours 0.835 (AP 0.519 / 0.492 / 0.832; Acc 0.555 / 0.464 / 0.741) | CONFIRMED |
| 11 | TrueFake Table IV | https://arxiv.org/abs/2504.20658, Table IV | SDXL, FB, TPR block: 0.21 (delta -0.79) in the fourth detector column (NPR per pass 1; column header not re-read here); FFHQ FB TNR 0.41 (-0.58) (MISLNet per pass 1) | CONFIRMED (pass 1 + grep) |
| 12 | RINE Table 4 | https://arxiv.org/abs/2402.19091, Table 4, "all generators" block, AVG column | w/o intermediate (last layer only, "equivalent to the SotA method [16]") 80.8 ACC / 94.2 AP; full 89.8 / 98.7. Difference = +9.0 ACC. Non-GAN generators block: 75.5 vs 86.6. Abstract/Intro claim "+9.4%" is a different comparison (vs state of the art, 1-class training), not this ablation. | CONFIRMED |

## Totals
12 items (17 sub-claims): 17 confirmed, 0 different, 0 not found.

## Caveats
- Item 1: the 28 vs 38 inconsistency is in the paper itself (Table 6 caption "38 sub-testsets" vs text "28 generation models"); cite as 92.2% "reported over 28 models / 38 sub-testsets".
- Item 6: the "17.4%-91.3%" range holds only for the five detectors that look safe at the ProGAN threshold; do not generalize to all 12 detectors.
- Item 7: 0.987 / 89.2% is the 384-px (High res.) model on the authors' own Comprehensive set, not a third-party benchmark.
- Item 9a: "0.95 -> 0.62" is a K24-sourced transfer attack at eps = 16/255; the clean 0.95 comes from RAID Table 1 (pass 1).
- Item 12: do not conflate the ablation's +9.0 with the abstract's +9.4 (different comparison).
- Not checked this pass: venue claims (ICASSP, ECCV, IJCNN), which remain as in `verification-log.md`.
