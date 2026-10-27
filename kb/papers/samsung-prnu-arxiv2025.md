# Diagonal Artifacts in Samsung Images: PRNU Challenges and Solutions
- **id:** samsung-prnu-arxiv2025
- **Authors / venue / year:** Vazquez-Padin, Perez-Gonzalez, Martin-Del-Rio; arXiv Oct 2025 (v1); no venue listed
- **Links:** arXiv 2510.09509 · no code link seen
- **Family:** camera-forensics
- **Status:** partially-verified

## Summary (plain language)
Recent Samsung Galaxy S and A phones add a shared diagonal pattern in default (processed) mode, so different devices of one model get colliding PRNU fingerprints and false positives. Raw (PRO-mode) captures bypass the pipeline and restore reliable PRNU.

## How it works
Compare fingerprints from default-mode vs raw images; PCE threshold tau = 60; analyse multi-frame/HDR images by JPEG APP4 metadata tags.

## Training data
Table 1: 20 device entries (4 USA Galaxy S, 8 Global S, 8 A5x); reference counts such as 40 images (SD06, 20 PRO) or 20 flat-field default images (Sec. 3) [counts as read by WebFetch from an HTML table, labels partly inferred] .

## Reported results
FPR at tau=60 (Fig. 3b): default-mode fingerprint S9 0.95, S9+ 1.0, S10 0.69, S10+ 0.66; raw fingerprint 0 in all four; TPR for SD06 0.94 in both. Most Global S PCE values exceed 60; A5x PCE mostly above 10^4 (Sec. 3).

## Known weaknesses / failure modes
Raw capture is unavailable on mid-range A series and in most forensic cases; HDR multi-frame images remain hard (Sec. 4).

## Code + weights
No code link seen.

## Verified?
Partially: numbers from arXiv HTML via WebFetch summarisation; table labels partly inferred.

## Relevance to this project
- Direct evidence that modern phone ISP pipelines add NON-unique shared patterns, so a 'PRNU realness' signal is weaker on recent phones (our fresh-phone FPR target).
- VISION (2010-2015 phones) would not show this effect; we cannot treat VISION PRNU results as representative of today's phones.
