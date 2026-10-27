# Scalable Black-Box Model Attribution for Images
- **id:** rpa-arxiv2026
- **Authors / venue / year:** Asaf Livne, Amir Jevnisek, Shai Avidan (Tel Aviv Univ.) / arXiv preprint 2608.15652v2 (20 Sep 2026; venue not stated) / 2026
- **Links:** arXiv 2608.15652 · project page https://asaf-livne.github.io/RPA/ (no code repo or license listed)
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
A very small CNN on raw 64x64 RGB patches attributes images among 25-27 generators with 95-99% accuracy on caption-matched benchmarks, with no access to the generators' internals. Also gives open-set and corruption numbers.

## How it works
CNN over 64x64 raw patches, per-patch logits averaged (Sec. 3, Eq. 3); AdamW, 75 epochs, flip augmentation only (S1). Parameter count inconsistent (66M in Sec. 3.2 vs 5.9M in S1).

## Training data
DRAGON (25 generators; 750/250/400 train/val/test per class), OpenFake (27 generators, incl. DALL-E 3, Imagen, Midjourney, GPT Image 1), AEDR (8 open latent-diffusion models, built by authors), GenImage (9 generators, few-shot).

## Reported results
- Closed-set (Table 1): DRAGON 25-class 98.9%; OpenFake 27-class 95.0%; best baseline DE-FAKE 62.0% / 57.3%; 2.79 ms/image.
- White-box AEDR pairwise (Table 2): 99.5% (AEDR 95.1%, LatentTracer 70.3%).
- Corruption-augmented model (Sec. 4.3): 98.0% clean DRAGON; JPEG/blur/resize/crop 'degrade gracefully', values only in Fig. 3 (not extracted).
- Open-set 17 known/10 unseen: AU-OSCR 0.875 +/- 0.037, AUROC 0.896 +/- 0.033 (Sec. 4.4, S7).
- Head-only adaptation 17->27 classes (Table 3): original 93.3%, new 85.1%, all 90.3%. Cross-dataset: 96.6% (OpenFake->DRAGON), 86.8% (DRAGON->OpenFake). GenImage few-shot (Table 4): 52.0% (1-shot) / 72.4% (10-shot).

## Known weaknesses / failure modes
- Generation settings (guidance, steps, scheduler) unexplored; OpenFake settings unknown; adversarial robustness untested (authors).
- Reviewer: fetch summary flags inconsistencies (66M vs 5.9M params; 9-way vs 99-way few-shot head).
- High accuracy on DRAGON may exploit format/size cues of the dataset (not tested here).

## Code + weights
No code or license listed.

## Verified?
Partially: arXiv HTML fetch summary; 2026 preprint, not peer-reviewed; Fig. 3 not read.

## Relevance to this project
- DRAGON uses the same captions and default settings for all models (Sec. 4.1), the only large benchmark besides WILD with a content control; accuracy there is high even for a tiny patch CNN, suggesting low-level generator traces are strong on clean data.
- Dataset-to-dataset transfer drops 96.6 -> 86.8: dataset-specific cues matter. Laundering numbers must be checked in Fig. 3.
