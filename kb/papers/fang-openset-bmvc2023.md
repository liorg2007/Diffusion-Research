# Open Set Synthetic Image Source Attribution
- **id:** fang-openset-bmvc2023
- **Authors / venue / year:** Shengbang Fang, Tai D. Nguyen, Matthew C. Stamm (Drexel) / BMVC 2023 / 2023
- **Links:** arXiv 2308.11557 · code: not found in fetched text
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Open-set source attribution via metric learning: an embedding network pretrained on camera-model identification is fine-tuned so each known generator has a centroid, and a test image is accepted only if close enough to its nearest centroid.

## How it works
MISLNet embedding, pretrained on 70 camera models, fine-tuned with ProxyNCA++ and L2 distance (Sec. 3.2, 4.4); nearest reference + normalised-distance threshold tau (Sec. 3.3-3.4). Random 256x256 crops, random JPEG QF in {75,80,85,90,95} or none (Sec. 4.1).

## Training data
7 generators, 10,000 images per generator/dataset pair (8k/1k/1k). Seen: ProGAN, Projected-GAN, StyleGAN2, StyleGAN3, Taming Transformer. Unseen: StyleGAN, Stable Diffusion (the only diffusion model).

## Reported results
- Table 3: seen average F1 0.900 (ProGAN 0.744, ProjGAN 0.974, SG2 0.875, SG3 0.969, Taming 0.940); unseen CRR StyleGAN 0.484, Stable Diffusion 0.806, avg 0.645.
- Table 2 (AUC of aF1-CRR curve): MISLNet+CamID 0.868 vs 0.761 from scratch.
- Baselines (aF1 / CRR): ResNet-50 0.867 / 0; RepMix 0.791 / 0; FSM 0.200 / 0.637.
- No separate robustness evaluation; closed-set accuracy not reported.

## Known weaknesses / failure modes
- Only 5 seen, 2 unseen generators; single run, no CIs.
- aF1 vs CRR trade-off via threshold.
- Generators trained on different content domains and SG overlaps seen training sets (reviewer inference from Table 1); no content control.
- Only one diffusion model, with unknown training data.

## Code + weights
Not found in fetched text; license unknown.

## Verified?
Partially: arXiv PDF text was not extractable by the fetch tool; numbers come from the arXiv HTML/ar5iv fetch summary.

## Relevance to this project
- Camera-forensics pretraining transfers to generator fingerprints: links to our camera-forensics line.
- Stable Diffusion rejected as unseen at 0.806 CRR but based on one SD model and no laundering test.
