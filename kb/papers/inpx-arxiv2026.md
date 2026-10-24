# AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange (INP-X)
- **id:** inpx-arxiv2026
- **Authors / venue / year:** Elif Nebioglu, Emirhan Bilgic, Adrian Popescu / arXiv preprint (no venue) / 2026
- **Links:** arXiv 2602.00192 (v1) · https://arxiv.org/abs/2602.00192 · code/data https://github.com/emirhanbilgic/INP-X
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
VAE encode-decode changes every pixel of an inpainted image, not just the edited region. The authors restore the original pixels outside the mask ("Inpainting Exchange") and show that many detectors collapse, i.e. they were detecting the global VAE re-encoding side effect (a spectral/high-frequency shift), not the generated content. This is direct evidence that the VAE trace is a strong detectable signal, and that it is erasable.

## How it works
Diagnostic dataset: inpaint with Kandinsky 2.2, OpenJourney, SD v1.4, then paste original pixels back outside the mask (INP-X). Evaluate 11 pretrained open-source detectors, 2 commercial APIs (N=1,000 each), and 4 fine-tuned backbones. Analysis of the SD1.4 VAE on real images, no diffusion step: image-level correlation between VAE reconstruction loss and high-frequency content r = 0.941, with inpainting loss r = 0.600 (Sec. 3.2/4.6); pixel-level VAE vs HighFreq r = 0.52 +- 0.08. Theorem 3.2 (variance contraction). Claim that the effect persists in SDXL and FLUX.1 is in Appendix A.10, which was not read **[unverified]**.
- **Weights needed:** none for the finding (analysis uses the SD1.4 VAE); it is a benchmark/diagnostic.
- **Passes per image:** n/a (diagnostic).

## Training data
Fine-tuning experiments: ResNet-50 / EfficientNet / CLIP ViT-B/32, B/16 trained on INP or INP-X (Table 3).

## Reported results
- Accuracy standard inpainting -> INP-X (Table 1): Corvi2023 0.942 -> 0.554; SPAI 0.661 -> 0.542; DNF 0.710 -> 0.604; UnivFD 0.552 -> 0.550. Commercial (Table 2): Hive 0.914 -> 0.548 (recall 0.834 -> 0.102), Sightengine 0.926 -> 0.550 (recall 0.862 -> 0.120). ResNet-50 trained on INP: 0.973 -> 0.557 on INP-X; trained on INP-X: 0.712 (Table 3).
- **Unseen-generator results:** generators limited to SD1.4, Kandinsky 2.2, OpenJourney inpainting; FLUX/SD3 not used in the dataset.
- **Laundering results:** Table 4, quality-80 JPEG: Corvi2023 0.803, Sightengine 0.831, Hive 0.885; blur (sigma 3): 0.768 / 0.724 / 0.905 (accuracy on INP-X-style data as read; exact condition **[unverified]**). Resize not tested.
- **FPR on real photos:** not reported (precision/recall only). Dataset normalized to JPEG (App. A.7).

## Known weaknesses / failure modes
Only VAE-based architectures covered; exchange can itself be used to evade detectors; commercial APIs on only 1,000 images; dataset count inconsistent (90K vs 30k per subset). Last 14.6k chars not read.

## Code + weights
- **License:** paper CC BY 4.0; code/data license not stated in fetched text **[unverified]**.

## Verified?
Partially: arXiv HTML v1, first 100k of 114.6k chars; Tables 1-5.

## Relevance to this project
- Strongest evidence in this batch that "diffusion detectors" are largely VAE-trace detectors, which supports a VAE-only approach for LDM-family generators.
- Same fact is a warning: a trace that is global and spectral can be removed or mimicked; test after resize/JPEG, and test on pixel-space/non-VAE generators separately.
