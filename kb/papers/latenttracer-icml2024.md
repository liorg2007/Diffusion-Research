# How to Trace Latent Generative Model Generated Images without Artificial Watermark?
- **id:** latenttracer-icml2024
- **Authors / venue / year:** Zhenting Wang, Vikash Sehwag, Chen Chen, Lingjuan Lyu, Dimitris N. Metaxas, Shiqing Ma / ICML 2024 (arXiv abstract page) / 2024
- **Links:** arXiv 2405.13360 · https://icml.cc/virtual/2024/poster/33958 · code https://github.com/ZhentingWang/LatentTracer
- **Family:** attribution
- **Status:** partially-verified

## Summary (plain language)
Asks "did THIS model make this image?" without any watermark. The image is pushed back into the model's latent space by gradient descent through the decoder, starting from the encoder's output. If the decoder can rebuild the image almost perfectly from the found latent, the image "belongs" to that model. It is the gradient-based cousin of AEROBLADE (which only does one encode/decode pass).

## How it works
Latent a0 = E(x); then gradient descent on the reconstruction loss of D(a) vs x, default 100 steps (about 400 with random init); encoder init starts much closer to the true latent (initial distance 0.019 vs 1.930 for SD v1-5, Table 1). Threshold per inspected model from N=100 of its own outputs, Grubbs-style test, alpha=0.05 (Eq. 1). Training-free, but white-box: needs the encoder AND decoder of every candidate model.

## Training data
None. Evaluation: 50 PromptHero prompts x 10 seeds per model (as stated in limitations of the fetched text); real LAION images for the "vs real" test.

## Reported results
- Belonging vs other model (Table 2): average 93.4%, range 89.2-98.2% (e.g. SD v1-4 vs SD v2-base 92.7%, SD v2-base vs v1-4 98.2%). RONAN baseline ~50% (48.8-55.6%).
- Belonging vs real LAION (Table 3): 97.4/97.8/98.1/98.3% for SD v1-4/v1-5/v2-base/Kandinsky, avg 97.9%. AUROC 0.96-0.99 (Table 7).
- Other autoencoders (Table 5): VAE 98.2%, VQ-VAE 97.5%, ViT-VQGAN 98.4%.
- Perturbed (Table 9, SD v2-base vs SD v1-5): JPEG Q95 96.5, Q90 95.2, Q80 94.1, Q70 90.8, Q60 90.2, Q50 89.3%. Gaussian noise std 0.01 97.4 -> 0.05 80.7%. Blur box 1 96.3, 2 88.7, 3 78.4%. Resizing: not reported in the text read. Cropping: listed, numbers not read.
- Cost (Table 4): 21.5-30.3 s per image with encoder init on an A100 (87.6-120.7 s random init).
- Fixed-FPR on phone photos: not reported.

## Known weaknesses / failure modes
- Needs white-box encoder+decoder; models sharing one autoencoder (SD v1-4/v1-5) cannot be separated (excluded; future work).
- Strong post-processing degrades it; failure cases: false positives with high brightness, false negatives with high shape complexity (authors).
- AEDR (aedr-aaai2026, Tables 1-2) reports only 70.4% / 66.7% average for LatentTracer on its 8-model benchmark, vs 93.4% / 97.9% in the original paper, and 43-210x slower.

## Code + weights
Repo: SD v1-5 and v2-base supported in README; Python 3.10; no LICENSE visible on the repo page **[unverified]**. Uses public HF Stable Diffusion checkpoints.

## Verified?
Partially: numbers from a fetch-tool summary of arXiv HTML (Tables 1-5, 7, 9); PDF not read directly.

## Relevance to this project
- Direct comparison point for "did this image pass through VAE X": the strongest white-box signal but ~20 s/image and JPEG-Q50 still 89% on a two-model pair.
- Not a detector of unseen generators; only answers membership for models we hold.
