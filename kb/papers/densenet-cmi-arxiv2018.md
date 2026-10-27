# Application of DenseNet in Camera Model Identification and Post-processing Detection
- **id:** densenet-cmi-arxiv2018
- **Authors / venue / year:** arXiv 1809.00576 (2018); authors and venue not retrieved [unverified]
- **Links:** arXiv 1809.00576 (ar5iv) · no code located
- **Family:** camera-forensics
- **Status:** read-abstract-only

## Summary (plain language)
Representative of learned camera-model identification: a pretrained DenseNet fine-tuned on image patches classifies which camera model took a photo. Reported near-perfect closed-set accuracy on Dresden, which shows that model-level traces are easy in-distribution.

## How it works
Transfer learning from DenseNet, multi-scale patches of 64, 128 and 256 pixels, closed-set classification (search-result summary of the abstract).

## Training data
Dresden (19 models) and IEEE SP Cup 2018 data (search summary) [unverified].

## Reported results
Over 99% test accuracy on Dresden, 19 models; 98.37% on IEEE SP Cup 2018 (from a search-result summary of the abstract, not table-verified) [unverified]. Another paper surfaced by the same search (Expert Systems with Applications 2022, homogeneous-patch hierarchical CNN) claims 99.01% 5-fold on Dresden natural [unverified].

## Known weaknesses / failure modes
Closed-set only; Dresden uses few devices per model and the same scenes shot by the same cameras, so patch-level scene/device leakage inflates numbers (general concern; not shown in this paper). No cross-dataset or social-media test in the summary.

## Code + weights
No code located.

## Verified?
No. Abstract summary via web search only.

## Relevance to this project
- Shows the in-distribution ceiling for model ID; does not tell us about VISION-like shared copies.
- Same lesson as AI-image detectors: closed-set 99% means little for generalisation.
