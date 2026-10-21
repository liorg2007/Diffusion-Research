# Verification log (primary-source PDFs, 2026-10-21)

PDFs downloaded from arxiv.org/pdf/<id> into data/papers_pdf/ (git-ignored); text extracted locally and table cells read directly.

| # | Claim | kb file | Source (paper, table, PDF page) | Result | Note |
|---|-------|---------|--------------------------------|--------|------|
| 1a | NPR on FakeInversion: original AUC 91.6 / bAcc 87.0; social-sim AUC 43.3 / bAcc 49.9 | bfree-cvpr2025.md | B-Free arXiv 2412.17671, Table 11, p.13 | confirmed | NLL/ECE 4.96/.123 and 27.1/.501 also match. 49.9 is also NPR's SynthWildX bAcc in Table 7 (different cell). |
| 1b | Simulation = resize factor 0.7-1 + JPEG QF 70-100 | bfree-cvpr2025.md | B-Free, appendix text, p.12 | confirmed | Quote matches. |
| 1c | Table 5/7 numbers (NPR, UnivFD, FatFormer, RINE; B-Free 96.3/0.10, 90.4/0.24, WildRF 95.6/86.2/97.3, +20.7) | bfree-cvpr2025.md | B-Free Tables 5 (p.7), 7 (p.8) | confirmed | NPR bAcc 0.6 (DALL-E 3), 0.4 (Firefly). Table 5 avgs NPR 42.7, UnivFD 62.4, FatFormer 52.9, RINE 65.0; Table 7 avgs 62.6/56.9/58.2/63.0. Added to kb. |
| 2 | NPR 45.4, LGrad 65.9, SPAI 91.0 avg AUC over 13 generators | spai-cvpr2025.md | SPAI arXiv 2411.19417, Table 1, p.6 | confirmed | Side finding CORRECTED: real sources are RAISE (DSLR), FODB (smartphone), ImageNet, COCO, OpenImages (not "three more smartphone sources"). |
| 3a | DINOv3 linear probe 97.5 real / 91.4 fake | vfm-baseline-arxiv2025.md | arXiv 2509.12995, Table 4, p.3 | confirmed | Avg 94.5. |
| 3b | WebAIG-25 size/composition | vfm-baseline-arxiv2025.md | arXiv 2509.12995, Sec 3.1, p.3 | confirmed (composition); size not stated in paper | Fakes: r/midjourney posts after model cut-offs; reals: private campus photos. No image count or cutoff date anywhere in the 5-page PDF; [unverified] tag replaced by "not stated". |
| 3c | CLIP-family real accuracy 27-47% | vfm-baseline-arxiv2025.md | Table 4, p.3 | confirmed with caveat | CLIP 46.7, SigLIP 31.3, Meta CLIP 36.2, Meta CLIP-2 27.4, PE 44.4; SigLIP-2 is 69.5 (outside range). |
| 4a | UnivFD (O23) AUROC 0.95 -> 0.50 under transfer attack | raid-arxiv2025.md | RAID arXiv 2506.03988, Tables 1-2, p.6 | corrected | CORRECTED: was 0.95 -> 0.50 (transfer from K24), PDF says 0.62 (F1 0.41). The 0.50 is the white-box diagonal; N-model ensemble gives 0.54. Other transfer cells (0.56, 0.51, 0.59) correct. |
| 4b | Epsilon | raid-arxiv2025.md | RAID Sec 2.1 (p.3), Tables 2-3 | corrected | CORRECTED: was "released dataset 8/255"; PDF: dataset has 24,000 adv images for each of 8/255, 16/255, 32/255; Table 2 = 16/255, Table 3 = 32/255; PGD 10 steps, step 0.05. |
| 5a | NPR TPR 0.21 on SDXL after Facebook | truefake-ijcnn2025.md | TrueFake arXiv 2504.20658, Table IV, p.6 | confirmed | Delta -0.79. MISLNet FFHQ-FB TNR 0.41 (-0.58) also confirmed. |
| 5b | JPEG QF per platform | truefake-ijcnn2025.md | TrueFake Table II, p.4 | confirmed | Facebook 61-92, X 87, Telegram 85; resize caps FB w>720, X w>1200, Telegram h>800. |
| 5c | Authors | truefake-ijcnn2025.md | TrueFake p.1 | confirmed | Stefano Dell'Anna, Andrea Montibeller, Giulia Boato (Trento; Boato also Truebees srl). |
| 5d | Venue IJCNN 2025 | truefake-ijcnn2025.md | PDF + arXiv abstract page | inaccessible (not stated in primary source) | No venue line in PDF or arXiv page; only repo name TrueFake-IJCNN25. Left tagged [venue unverified]. |
| 6a | 12 detectors; ~9.1k phone photos | laion-mobile-arxiv2026.md | LAION-Mobile arXiv 2609.11134, abstract p.1, Sec 3 p.4 | confirmed | 10,000 IDs requested, 9,171 retrieved, 9,115 scored (DIRE 738). |
| 6b | 5/12 detectors flag 17-91% of reals at EER threshold | laion-mobile-arxiv2026.md | Table 4, p.7 | confirmed | Five detectors (<=11% FPR at ProGAN threshold) go to 17.4 / 39.1 / 42.9 / 89.3 / 91.3% at NTIRE-EER threshold. Abstract says "several"; "5" derived from the table. |
| 6c | 0.03% post-2021 phones | laion-mobile-arxiv2026.md | Sec 3, p.4 | confirmed | 212 images. |
| 6d | Dataset license | laion-mobile-arxiv2026.md | Hugging Face API (laionmobile/laion-mobile) | confirmed via HF metadata (not in PDF) | License tag cc-by-4.0, not gated, metadata-only (no images). PDF states no dataset license. |
| 7 | ~+11 pt cross-generator accuracy | fakeorjpeg-arxiv2024.md | arXiv 2403.17608, Table 4, p.10 | confirmed with caveat | ResNet50 71.68 -> 82.74 (+11.06), Swin-T +11.74. The gain is from JPEG96 + size-constrained training combined; JPEG-only de-biasing helps only marginally (Sec 4.1). |
| 8 | Grag2021 resize+JPEG: AUC ~91 / Acc 61.5 | corvi-icassp2023.md | arXiv 2211.00680, Table 1, p.4 | confirmed | 61.5/90.8 (Acc/AUC); uncompressed 75.8/92.8. ICASSP venue not printed in PDF text (still unverified). |
| 9 | Chameleon: 94-100% real acc, 0-27% fake acc | aide-chameleon-iclr2025.md | arXiv 2406.19435 (ICLR 2025 header), Table 5, p.9 | confirmed for AIDE/NPR; not for all detectors | AIDE real 94.4-98.5, fake 5.0-26.8. Not universal: UnivFD fake 75-86 / real ~41, FreDect real 87.1, PatchCraft real 92.8. |

Totals: 20 sub-claims checked; 14 confirmed (incl. 3 with caveats and 1 via HF metadata), 3 corrected (RAID O23 transfer value, RAID epsilon, SPAI real-source list), 1 partially (WebAIG-25 size not stated in paper), 1 inaccessible (TrueFake venue not in primary source).
