# Decoder-bank universality: a trace direction learned from real-vs-round-trip pairs of several unrelated autoencoders transfers to a held-out decoder family and to closed generators
- **id:** decoder-bank-universality
- **Gap addressed:** generalisation of decoder-trace detectors to UNSEEN decoders (FLUX/SD3 16-channel VAE, DALL-E 3, Midjourney, Firefly) without any generator images
- **Scores (1-5):** novelty 2 · expected gain on unseen/in-the-wild 3 · feasibility 4 · **product 24** (rank 5 in docs/06-diffusion-hypotheses.md)

- **Hypothesis:** A frozen-DINOv2 (F2) probe trained on real-vs-own-round-trip pairs pooled over K >= 2 autoencoders separates a held-out autoencoder's pairs at pair AUROC >= 0.85 (fov512, clean), and raises transfer to non-SD Synthbuster generators by >= 0.03 macro AUROC over the SD-VAE-only probe (0.875 fov512 clean, 011/pilot/tables.md), i.e. latent decoders share a common trace.

- **Mechanism (why diffusion images differ):** all current latent generators render pixels through a convolutional decoder from a low-resolution latent (stride 8 for SD/SDXL/FLUX; MoVQ for Kandinsky). They share an information bottleneck that attenuates and re-synthesises fine texture (inpx-arxiv2026: VAE loss vs high-frequency content r = 0.941) and upsampling stages. If the DINOv2 trace direction captures the shared part, it should hold across decoders; if it captures SD-VAE idiosyncrasies, it will not.

- **Closest prior work:** aligned-vae-frozen-probe (our #6; Rajan et al. ICLR 2025, arXiv 2410.11835; DDA NeurIPS 2025, arXiv 2505.14359), aeroblade-cvpr2024 and hfi-arxiv2024 (min over an AE bank, training-free), drct-icml2024. A web search snippet (DDA PDF, not re-opened) says detectors trained on reconstructions from different VAEs are "broadly comparable"; arXiv 2605.14486 (cited in #6) argues single-VAE negatives are too homogeneous. Difference: explicit leave-one-decoder-out test of the shared-trace claim on held-out decoders and closed generators, with the E4 pair metric. Mostly an evaluation contribution.

- **Evidence for / against:**
  - For: the SD-VAE-only F2 probe already reaches 0.875 on DALL-E 2/3, Firefly, MJ5 in fov512 clean (011 transfer table); AEROBLADE used the SD2 AE as a proxy for Midjourney successfully (aeroblade-cvpr2024).
  - Against: LDM-trained detectors fail on DALL-E 2/3 and Firefly (corvi-icassp2023 Table 2; bfree-cvpr2025 Table 5 via DMID: FLUX 63.7 bAcc); the trace is erased by laundering (E4 0.60), so this addresses only the unseen-generator axis; SDXL VAE failed the D-008 GPU guard and must run on CPU (about 11 s/pass, 009/pilot/critique.md).

- **Test plan:**
  - *Pilot now (no download):* SD-VAE (GPU) and SDXL-VAE (CPU) round-trips of 99 train natives + 120 test natives per arm (SDXL: ~440 passes, about 1.4 h CPU). Probes: P_SD (existing E4 F2), P_XL, P_SD+XL. Report cross-decoder pair AUROC (P_SD on SDXL pairs and vice versa) and transfer to the 6 generators, clean and laundered, fov512 + orig, with texture baseline.
  - *Official (downloads):* add Kandinsky 2.2 MoVQ (Apache-2.0, not gated, 271 MB) as a third, architecturally different decoder; hold out the FLUX VAE (gated click-through, 168 MB; owner approval per D-015) as an oracle-only arm; test on held-out FLUX/SD3.5 generator images and RAISE reals.
  - **Pass:** cross-decoder pair AUROC >= 0.85 fov512 clean (both directions); bank probe non-SD transfer >= SD-only + 0.03 clean with no loss laundered; beats texture by >= 0.10. **Fail:** cross-decoder pair AUROC < 0.70 (trace is decoder-specific).
  - *Weights:* have sd-vae-ft-mse + sdxl-vae; later MoVQ 271 MB (https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder, licence apache-2.0, gated False per the HF API) and optionally FLUX VAE 168 MB (https://huggingface.co/black-forest-labs/FLUX.1-schnell, apache-2.0 but gated).
  - *Compute:* pilot ~1.5 h CPU + minutes GPU; official ~3-4 h CPU for SDXL [estimate].
  - *Phase-5 reach:* NO alone (does not address laundering); it decides which decoder bank self-ref-trace-contrast should use.

- **Status:** proposed
