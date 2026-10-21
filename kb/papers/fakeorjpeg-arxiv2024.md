# Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets
- **id:** fakeorjpeg-arxiv2024
- **Authors / venue / year:** Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt, Janis Keuper. arXiv preprint March 2024 (v2); no venue listed on arXiv page.
- **Links:** arXiv 2403.17608 · https://arxiv.org/abs/2403.17608 · data/code claimed at https://www.unbiased-genimage.org (anonymous site; availability not checked **[unverified]**)
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
In GenImage, real photos are JPEG and fakes are PNG, and real images cluster in a narrow size range, so detectors can score "JPEG-ness" or size. Removing these biases makes detectors much more JPEG-robust and changes cross-generator results.

## How it works
Analysis plus re-built GenImage variants (JPEG-matched, size-matched) and retraining ResNet50 / Swin-T.

## Training data
GenImage (see genimage-neurips2023); "most natural GenImage images have both height and width within [450, 550]" (Sec 4.2); "the majority of images [are] compressed using a quality factor of 96" (Sec 3.1, Fig 6 - log-scale, no exact share).

## Reported results
- Table 2 (ResNet50 trained on Midjourney subset, tested on natural 1024x1024 FFHQ): accuracy PNG 80.45; JPEG95 94.84; JPEG90 98.93; JPEG80 99.95; JPEG70 99.99; JPEG60 100.0 -> detector flags compression as "natural".
- Table 3 (avg accuracy over generator matrix, classic vs JPEG96-only training): JPEG95 53.91 -> 67.17 (+13.26); JPEG80 50.62 -> 59.37 (+8.75); JPEG60 50.58 -> 55.07 (+4.49).
- Size-constrained training: samples ~320,000 -> ~75,000; max gain 41.29 pp, average +11.06 pp (Sec 4.2).
- Table 4 cross-generator avg: ResNet50 71.68 -> 82.74 (+11.06); Swin-T 74.09 -> 85.83 (+11.74).

## Known weaknesses / failure modes
Authors' fix still uses GenImage's old/low-res generators; no social-media or phone-photo evaluation. Not peer-reviewed.

## Code + weights
Site referenced above; licence not verified **[unverified]**.

## Verified?
Numbers from arXiv HTML v2 via fetch tool (not PDF-checked).
(verified vs PDF p.10, 2026-10-21) Table 4 confirmed: ResNet50 71.68 -> 82.74 (+11.06), Swin-T 74.09 -> 85.83 (+11.74); max +41.29 pp; 320k -> 75k samples. Caveat: the "Ours" column is JPEG96 AND size-constrained training combined (Sec 4.2); the paper says JPEG-only de-biasing improves cross-generator results only marginally. Tables 2-3 (p.6, p.8) also match.

## Relevance to this project
Direct warning: in our eval set, real and fake must share container format, JPEG quality history and resolution, otherwise measured accuracy is partly format detection.
