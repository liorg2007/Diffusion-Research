# Representation and Reference Selection in Training-Free Synthetic Image Attribution
- **id:** repsel-arxiv2026
- **Authors / venue / year:** Meiling Li, Pietro Bongini, Benedetta Tondi, Mauro Barni / arXiv preprint (v1, 13 Jul 2026; no venue listed) / 2026
- **Links:** arXiv 2607.12052 · https://arxiv.org/abs/2607.12052 · code not checked
- **Family:** attribution
- **Status:** partially-verified (arXiv abstract page + HTML text via fetch summary; PDF not read directly)

## Summary (plain language)
Controlled study of which frozen-encoder layer, and which reference images, work best for attributing an image to its generator by comparing it with reference sets, with no trained classifier. Main finding: accuracy peaks at intermediate layers of both CLIP and DINOv2 and falls at the shallow and final layers.

## How it works
Frozen CLIP / DINOv2 features from different layers (CLS token; preliminary comparison said CLS beat patch pooling), nearest-reference comparison. Three reference strategies: arbitrary, semantically aligned, resynthesis-based. Linear probes on CLIP are trained only as diagnostics.

## Training data
None for the attributor (training-free). Datasets include BC-Attr-6 (details not verified).

## Reported results
- Sec. V-A, Fig. 3: "both encoders perform best at intermediate layers, while accuracy drops at both shallow and final layers" (plot; no DINOv2 numbers in text).
- Table II (CLIP): L12 best in every dataset/reference setting; e.g. BC-Attr-6, M=10, semantically aligned: 56.67 (L6), 85.33 (L12), 29.25 (L24).
- Fig. 4: probes show semantic separability rising with depth while generator separability peaks mid-depth.
- Fig. 5: under JPEG / blur / resize "accuracy still peaks at intermediate CLIP layers and drops toward the final layer"; resize mildest, JPEG and blur worse as severity rises (no numbers in text). **[unverified]** against the figure itself.

## Known weaknesses / failure modes
No patch-shuffle or texture analysis; no comparison with detector embeddings; perturbations applied to queries at test time with a training-free method, so no train-time laundering question arises.

## Code + weights
Not checked.

## Verified?
Partially: abstract page read; body quotes come from a fetch-tool summary of the arXiv HTML. Figures not inspected.

## Relevance to this project
- Closest prior art for claim C's "mid-depth beats last layer for attribution, also after JPEG/resize". Our novelty is only the trained-probe, laundering-augmented, held-out-laundering setting and the 8-way size-controlled data.
