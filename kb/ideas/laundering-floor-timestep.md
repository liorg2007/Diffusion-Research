# Laundering-floor timestep: evaluate the diffusion denoiser at a noise level just above the laundering residual, where JPEG/resize damage is drowned but the generator's density bias is still visible
- **id:** laundering-floor-timestep
- **Gap addressed:** denoiser/trajectory detectors (TSG, LaRE2, DTAD, ESIDE, LATTE) are untested or near random under JPEG+resize; timestep choice is unprincipled
- **Scores (1-5):** novelty 3 · expected gain on unseen/in-the-wild 3 · feasibility 3 · **product 27** (rank 4 in docs/06-diffusion-hypotheses.md)

- **Hypothesis:** With the SD1.5 UNet (empty prompt), the single-step noise-prediction error at a timestep t* chosen BEFORE seeing labels so that the latent noise std sigma_t is 2-4x the measured laundering residual in SD-VAE latent space loses <= 0.05 macro AUROC from clean to jpeg75_resize05 on 6 unseen Synthbuster generators, while the same score at t <= 50 (TSG-style) loses >= 0.15.

- **Mechanism (why diffusion images differ):**
  - Diffusion samples lie where the learned score is small: the denoiser predicts the added noise better on them (DIRE, LaRE2, manifold-bias, DTAD all build on this; tsg-arxiv2024 uses eps_theta at t = 0).
  - Laundering adds a perturbation delta of roughly fixed size in latent space. At small t (sigma_t << |delta|) the denoiser mainly "sees" delta, which is the same off-manifold JPEG/resize damage for reals and fakes, so the class gap disappears (independent evidence: DTAD near random under high JPEG, dtad-neurips2025 via Sonoda Fig. 4, figure only). At sigma_t >> |delta| the added Gaussian noise drowns delta, and the prediction error reflects coarser structure where the generator's density bias remains. Too large t erases the image. Hence a laundering-dependent optimum t* predicted from |delta| alone.
  - LaRE2 already found t in [150, 300] stable on clean GenImage (lare2-cvpr2024 Fig. 6) - consistent with, but not a test of, this.

- **Closest prior work:** tsg-arxiv2024 (t = 0/50, JPEG only in a figure), lare2-cvpr2024 (t = 200, no JPEG results), latte-arxiv2025 (5 timesteps, weak under strong JPEG), eside-arxiv2025 (timestep ensemble, JPEG/resize untested), dtad-neurips2025, manifold-bias-iclr2025 (JPEG AUC 0.835 -> 0.79, Table 2; no resize). Web: ANL (arXiv 2604.14570, search snippet: single-timestep noise estimation, timestep strategy in a figure) and TSG++ (MMAsia 2025 PDF, search snippet: multi-timestep fusion "improves robustness"); neither opened. No paper found that sets t from a measured laundering floor or tests JPEG+resize this way.

- **Evidence for / against:**
  - For: the mechanism is simple and its prediction (location of t*) is set before labels.
  - Against: needs a 1.72 GB UNet; the score may carry content/texture bias like other recon scores (#7 lost to texture, docs/05 §2); UNet tied to SD1.5 may not transfer to FLUX/DALL-E 3 (TSG gives no FLUX/DALL-E 3 results); white-box fragile (fragile-recon-arxiv2026).

- **Test plan:**
  - *Pilot now (no download):* measure the laundering floor: |E(L(x)) − E(x)| per latent element with sd-vae-ft-mse for 358 natives and 720 fakes, for jpeg75_resize05 and social_sim (L(x) upsampled back to the encoder size for comparability), and map it onto the SD1.5 DDPM schedule (sigma_t = sqrt((1 − alphabar_t)/alphabar_t) in the scaled latent, schedule from the public scheduler config) to pre-register t* and two control timesteps (t = 50 and 3·t*). Report whether |delta| differs between classes (if it does, it is itself a cue and must be controlled). About 15 min GPU [estimate].
  - *Main pilot (after download approval):* SD1.5 UNet fp16, empty-prompt embedding (SD1.5's text encoder is CLIP ViT-L/14 text; whether our local openai-clip-vitl14 weights reproduce it is [unverified] - otherwise also fetch text_encoder fp16, size [unverified]). Score = mean ||eps_theta(x_t, t) − eps||^2 over 4 noise draws at t in {50, t*, 3t*}; same images/arms/conditions as above; probe-free (training-free) and a logistic variant on the error map pooled stats trained on LOGO folds.
  - **Pass:** at t*, macro AUROC drop clean -> laundered <= 0.05 and laundered AUROC >= 0.80 (fov512), >= texture + 0.10; at t = 50 drop >= 0.15; FPR <= 2% on WA/FBH at the natives' 1% threshold. **Fail:** laundered AUROC at t* < 0.70 or t* not better than t = 50 under laundering by >= 0.05.
  - *Weights:* sd-legacy/stable-diffusion-v1-5 `unet/diffusion_pytorch_model.fp16.safetensors` 1.72 GB, CreativeML OpenRAIL-M (use-based restrictions; gating not shown on the page [unverified]) (https://huggingface.co/sd-legacy/stable-diffusion-v1-5/tree/main/unet, opened). Within the D-015 headroom; single item < 5 GB; licence to be noted in kb.
  - *Compute:* 1,794 test images x 3 conditions x 2 arms x 3 timesteps x 4 draws ~= 130k UNet forwards at 64x64 latent fp16; at an assumed 15-25 it/s batched on an RTX 4060 Ti [unverified] about 1.5-2.5 h. Fits in 8 GB at fp16 [estimate].
  - *Phase-5 reach:* possible but unlikely alone (no evidence yet of > 0.9 under laundering for any denoiser score); a candidate second channel to fuse with self-ref-trace-contrast.

- **Status:** proposed
