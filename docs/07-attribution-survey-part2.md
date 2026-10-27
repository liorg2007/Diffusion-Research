# Attribution survey, part 2: training-free / reconstruction-based source tracing
Scope: which model (or autoencoder) made an image, without training a classifier. Complements kb entries aeroblade-cvpr2024, lare2-cvpr2024, fire-cvpr2025, hfi-arxiv2024.
Grounding: numbers below are from kb/papers/*.md (each cites its table); sources were read through a fetch tool summary of arXiv HTML, not the PDFs, so treat as partially-verified.

## 1. Papers and what they report
| id | Needs | Reported accuracy | Cost / image | JPEG / resize |
|---|---|---|---|---|
| ronan-neurips2023 | white-box generator, any type | ~95.7% TPR at ~5% FPR (intro); SD2.0 case 87.5% on 20+20 imgs (Sec 5.3) | 55-153 s (appendix, unit unclear) | not tested |
| latenttracer-icml2024 | encoder+decoder of each candidate (SD1.4/1.5/2-base/SDXL/Kandinsky) | 93.4% vs other model (Table 2); 97.9% vs real LAION (Table 3) | 21.5-30.3 s, A100 (Table 4) | JPEG Q90 95.2, Q70 90.8, Q50 89.3% (Table 9); noise/blur degrade faster; resize not seen |
| aedr-aaai2026 | the candidate VAE only | 95.1% vs other models, 96.9% vs real (Tables 1-2); MoVQ 82.9% | 0.06-1.25 s (Table 3, hardware not stated) | none in paper |
| dna-arxiv2026 | one VAE per family + (stage 2) all variant backbones | stage 1 98.4% acc binary (Tab IV); 89.11% end-to-end (Tab VI) | stage 1 0.107 s; stage 2 1-87 s | JPEG QF75: 9.15% end-to-end; Facebook pilot 3% (Sec VII-E) |
| corvi-intriguing-cvprw2023 | nothing (spectra of residuals) | qualitative only | negligible | peaks fade under JPEG/resize (Fig 4, qualitative) |
| beyondpeaks-arxiv2025 | nothing | linear peak detector TPR 0.75-0.86 clean (Table 1), -0.83 rel. after peak removal | negligible | untested; authors expect failure under JPEG |
| sds-arxiv2026 | forward passes of candidate U-Nets | claims ~99.9% on 8 models (abstract only) | not read | not read |
| protoleaknet-arxiv2025 | SD2.1 VAE (SDXL VAE ~same) + trained ResNet-18 | closed AUC 98.13 (T1); post-proc 92-96 (T2); open-set AUC ~57 (T3) | not reported | mixed pipeline only |

## 2. Models, weights, licences
- AEDR/LatentTracer/RONAN need the TARGET model's autoencoder. SD1.x/2.x/SDXL/Kandinsky weights are public on Hugging Face; the exact licences were not read here [unverified]. FLUX.1-dev (in AEDR) has a non-commercial licence [unverified, not read]. Midjourney/DALL-E/Firefly VAEs are not public, so those cannot be attributed this way (AEROBLADE used SD2 AE as proxy, see its kb entry).
- Code licences: none of LatentTracer, AEDR, DNA, Beyond-Peaks repos showed a licence in the pages read [unverified]. Check LICENSE files before reuse.

## 3. Findings that matter for us
1. Lossless-only evidence. Only LatentTracer reports JPEG numbers (89-95% for Q50-Q90, a two-model pair). The only real-world test of the cheap AEDR signal (DNA, Sec VII-E) shows collapse at JPEG QF75 (9.15% end-to-end over 24 candidate models, random 0.40% per Tab VI, so above chance but unusable).
2. Gradient inversion (LatentTracer) is far more JPEG-tolerant than one-shot ratio signals but costs ~20-30 s/image on an A100, i.e. 43-210x AEDR (AEDR Table 3).
3. Shared VAE => unseparable. SD1.x/2.x share one VAE family; SDXL has its own. VAE-based attribution gives "VAE family", not "model" (LatentTracer limitation; DNA stage 2 and SDS exist for that reason and need U-Nets).
4. Independent comparison: AEDR reports LatentTracer at 70.4%/66.7% on its benchmark versus 93.4%/97.9% in the original paper (different protocols, see Tables 1-2 vs Table 2/3).
5. Spectral peaks (1/4, 1/8 multiples; Corvi) are produced by every f8 VAE decoder, so they cannot discriminate sd-vae-ft-mse from SDXL VAE, and are removed by JPEG/resize.

## 4. Feasible on an 8 GB RTX 4060 Ti with our two VAEs
Assumption to check: sd-vae-ft-mse is a fine-tuned DECODER of the original SD kl-f8 VAE; the VAE shipped inside SD1.5/SD2.x pipelines may differ from it [unverified, check the HF model cards before claiming "SD1.5's VAE"]. sdxl-vae is SDXL's own VAE.
- Feasible now (cost est. is ours, not from papers): AEDR-style double reconstruction and AEROBLADE-style one-pass LPIPS, with both VAEs, at 512 px in fp16: well under 1 s/image, a few hundred MB VRAM for a VAE [estimate, untested]. Output per image: two scores (one per VAE) plus calibrated threshold from ~500 own-model generations (AEDR Eq 6) -- needs us to generate/obtain such images (SD1.5 and SDXL pipelines would require extra weights beyond the VAEs; ask owner before downloading).
- Feasible but slow: LatentTracer-style inversion, 100 Adam steps through the decoder; memory for backprop through a 512 px VAE decoder in fp16 probably fits 8 GB, maybe with gradient checkpointing [unverified]; time est. 10-60 s/image, so use only on a small subset (hundreds of images).
- Not feasible: DNA stage 2, SDS (need full U-Nets of every candidate; FLUX.2 86 s/image); RONAN (slow, poor on LDMs).
- Cheap spectral add-on: Corvi-style residual spectra / peaks. Cost negligible, but expect no signal after jpeg75_resize05 and social_sim.

## 5. Suggested next experiment (for the lead to decide)
Test AEDR-style two-VAE scores (sd-vae-ft-mse, sdxl-vae) as FEATURES for "VAE-family attribution + detection" under our D-009 conditions (clean, jpeg75_resize05, social_sim). Prediction from DNA: clean high, jpeg75 near chance. Controls: AE-laundered real images (beyondpeaks Table 3) to check false-alarm behaviour; report FPR on fresh phone photos. A pass would be AUROC >= 0.95 after jpeg75_resize05; the literature gives no reason to expect it (fire-cvpr2025 and fragile-recon-arxiv2026 point the same way).
Risk: CLAUDE.md warns against in-distribution-only claims; attribution of known VAEs says nothing about unseen generators with unknown VAEs.
