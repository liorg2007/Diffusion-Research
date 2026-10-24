# Diffusion-specific hypotheses (D-014 refocus)

**Written:** 2026-10-24 by an opus/high subagent, for the lead to review and merge. **Inputs:** CLAUDE.md, DECISIONS.md (D-013 to D-015), docs/02-hypotheses.md (the N×G×F ranking), docs/05-phase4-pilot-synthesis.md, kb/index.md and the 13 diffusion papers plus older reconstruction entries, the pilot critiques in experiments/005–010, and the E4 results in experiments/011-vae-trace/pilot/tables.md. There was no E4 critique when this was written. Web checks are listed in §7.
**Status:** every hypothesis here is PROPOSED. Nothing was run for this document. No pilot counts toward the stop rule (plan Amendment 2).

## 1. What the evidence says, in one paragraph each

- **E4 shows the decoder trace is real, and the problem is content variance, not erasure.**
  - A probe separates a real photo from its own SD-VAE round-trip at a pooled pair AUROC of 0.859 (orig, F2) / 0.945 (fov512, F2) clean. Within a pair, the round-trip scores higher 99.1% / 100% of the time.
  - Under jpeg75_resize05 the pooled AUROC drops to 0.597 / 0.599 (probe trained on clean pairs). With a probe trained on laundered pairs it is 0.670 / 0.739, yet the within-pair win-rate stays at **0.980** in both arms.
  - (All numbers: experiments/011-vae-trace/pilot/tables.md.)
  - So laundering shrinks the trace below the spread *between* images, but the trace is still there *within* a pair. A detector that can build a same-content reference at test time could get much of it back. This drives H-D1.
- **E2 (#6):** a real-vs-round-trip probe transfers to unseen generators: 0.819 clean / 0.778 laundered. But it rides a "not straight from a camera" axis rather than the trace: unseen-device natives vs their own round-trips give only 0.66 / 0.56, and RAID reals give 0.52 (009/pilot/critique.md F1).
- **Pixel-space reconstruction scores track texture.** #7 scored 0.601 clean / 0.318 laundered and lost to a Laplacian texture baseline (0.778) (docs/05 §2). Zoom/texture confounds are a standing risk (D-013, docs/05 §3), so every hypothesis below carries a texture baseline and both arms.
- **The literature has gaps.**
  - No KB diffusion paper reports FPR on real photos, or numbers under JPEG plus resize. The robustness results are figures only: hfi, fire, latte, tsg, aeroblade.
  - DTAD is near random under JPEG (dtad-neurips2025, via Sonoda Fig. 4).
  - INP-X ties detection to the VAE re-encoding trace (inpx-arxiv2026).
  - Reconstruction detectors break under white-box attacks (fragile-recon-arxiv2026, Fig. 3).

## 2. Shared protocol for every pilot below

These rules are fixed before any result.

- **Data available now:** VISION train devices (10) and test devices (25), with native, WhatsApp (WA) and Facebook-HQ (FBH) copies; 6 Synthbuster generators × 120 (SD2, SDXL, DALL-E 2, DALL-E 3, Firefly, MJ v5); sd-vae-ft-mse on the GPU (D-008 passed); SDXL VAE on the CPU only; DINOv2-base.
- **Arms:** `fov512` is the label arm, following the docs/05 §5a recommendation (the owner still decides D-013). `orig` is reported with the same rules.
- **Conditions:** clean, jpeg75_resize05 and social_sim (experiments/lib/perturb.py), applied identically to both classes with EXIF stripped. Breakdowns: per generator, SD-family vs non-SD.
- **Gates:**
  - D-008 guard.
  - Laplacian texture baseline. A method must beat it by ≥ 0.10.
  - A mechanism-specific validity gate, listed per hypothesis. If the gate fails, the result is UNINFORMATIVE, not FAIL.
  - Direction fixed on training data, never flipped on test.
- **FPR:**
  - Threshold at 1% FPR, calibrated on held-out-device natives.
  - FPR reported on the other devices' natives and on WA and FBH copies, with a device bootstrap.
  - Pilot bar: ≤ 2%. The Phase-5 bar is ≤ 1% on fresh phones; this needs the H3 owner photos.
- **Official run (when grip.unina.it returns):** held-out FLUX and SD3.5 (16-channel VAEs, never in any bank or training set), RAISE PNG reals, ≥ 30 devices, comparators B0, B-Free, CommFor and RINE.

## 3. Ranked hypotheses

The scales follow docs/02:
- **N (novelty):** 1 = published, 3 = new combination or untested prediction, 5 = new mechanism.
- **G (gain):** expected effect on unseen generators, laundering and FPR.
- **F (feasibility):** on the RTX 4060 Ti 8 GB within the D-015 budget; 5 = runs now with nothing new to download.

| Rank | id | One-line hypothesis | N | G | F | N×G×F | Phase-5 route? |
|---|---|---|---|---|---|---|---|
| 1 | [self-ref-trace-contrast](../kb/ideas/self-ref-trace-contrast.md) | Score w·[φ(L0(VAE t)) − φ(L0(t))]. Real photos move along the learned trace direction under a VAE round-trip; decoded images barely move. Applying the same canonical laundering L0 to both sides keeps the band where this still holds. Target: ≥ 0.80 laundered (fov512), ≥ +0.10 over E4's pooled probe | 3 | 4 | 5 | **60** | **Plausible.** The only one with a route to ≥ 0.95 laundered: the within-pair ceiling is about 0.98. FPR ≤ 1% plausible with fov512 calibration |
| 2 | [latent-grid-periodicity-rescale](../kb/ideas/latent-grid-periodicity-rescale.md) | The 8-px latent grid survives non-dyadic downscaling plus JPEG (oracle AUROC ≥ 0.80 at s = 0.6/0.7/0.8) but collapses at s = 0.5 (≤ 0.65), because 0.5× maps it onto JPEG block harmonics | 3 | 3 | 5 | **45** | **No** at jpeg75_resize05, by its own prediction. A fusion feature for social_sim, and a protocol finding |
| 3 | [variance-contraction-patch-rank](../kb/ideas/variance-contraction-patch-rank.md) | The effective rank of DINOv2 block-8 patch tokens is lower for diffusion images, even after regressing out texture. Target: ≥ 0.75 laundered, with a drop ≤ 0.05 | 3 | 2 | 5 | **30** | **No** alone (high texture risk) |
| 4 | [laundering-floor-timestep](../kb/ideas/laundering-floor-timestep.md) | SD1.5 UNet noise-prediction error at a t* fixed beforehand from the measured laundering residual (σ_t ≈ 2–4× residual) loses ≤ 0.05 under laundering, while t ≤ 50 loses ≥ 0.15 | 3 | 3 | 3 | **27** | Possible but unproven. A second channel for fusion with #1 |
| 5 | [decoder-bank-universality](../kb/ideas/decoder-bank-universality.md) | A trace probe trained on pairs from ≥ 2 autoencoders separates a held-out decoder's pairs at ≥ 0.85, and adds ≥ 0.03 on non-SD generators | 2 | 3 | 4 | **24** | **No** alone (does not touch laundering). Decides #1's VAE bank |

Considered and dropped:
- Laundering-matched aligned training. E4's laundered-pair probe transfers worse to laundered fakes (orig F2 0.494, fov512 0.626, 011 transfer table).
- A sensor-noise (Poisson–Gaussian) "camera physics" score. It is not diffusion-specific, and modern neural-ISP denoising would likely raise FPR [unverified].

## 4. Per-hypothesis summary

The full mechanism, prior work, evidence, gates and pass/fail rules are in each kb/ideas file.

### H-D1: self-ref-trace-contrast (rank 1, score 60). Directly addresses laundering fragility
- **Mechanism:** a decoder output is nearly a fixed point of a VAE round-trip; a real photo is not (AEROBLADE/HFI premise).
  - Reading the shift along E4's DINOv2 trace direction avoids the texture trap of pixel errors. E4's direction beats texture by +0.31 / +0.41 clean.
  - Taking the difference against a same-content reference removes content variance, the cause of E4's pooled-vs-paired gap.
  - Applying the same L0 (jpeg75_resize05) to t and VAE(t) removes the fine band. In that band the VAE re-synthesises detail for laundered reals *and* fakes, which is what breaks idempotence.
- **Prediction (pilot, fov512):**
  - Arm B: ≥ 0.80 laundered (CI lower bound ≥ 0.75) and ≥ pooled F2 + 0.10 (pooled F2: 0.665 fov512 / 0.643 orig).
  - ≥ 0.90 clean; ≥ 0.80 social_sim; non-SD group ≥ 0.75 laundered.
  - FPR ≤ 2% on WA/FBH.
  - **Fail** if < 0.70 laundered, or ≤ pooled + 0.03.
  - Also arm A2, a generator-free probe on difference vectors: real (first round-trip) vs second round-trip.
- **Gates:**
  - Saturation: second-round-trip shift ≤ 0.5× the first-round-trip shift on natives.
  - Content cancellation: Spearman(w·φ(t), w·φ(VAE t)) ≥ 0.8.
- **Weights:** none new (sd-vae-ft-mse and DINOv2 are cached). Licence: the SD VAE is already recorded in env/SOURCES.md.
- **Compute:** about 8.6k VAE passes, about 1 h on the GPU. Basis: E4 ran 1,074 passes in 418–504 s (011/pilot/run.log).
- **Novelty:** closest are AEROBLADE/HFI (pixel/LPIPS, training-free), FIRE (mid-band recon error, trained, 68–73% ACC on unseen generators) and the CVPR 2026 Difference-in-Difference paper (arXiv 2602.23732; abstract only: a second-order difference of reconstruction errors to cut variance; space and laundering results not stated [unverified]). New here:
  - a feature-space projection learned from real pairs only;
  - matched canonical laundering on both sides;
  - a JPEG+resize evaluation with real-photo FPR.
- **Risk:** laundered fakes may lose idempotence completely; the trace direction may be SD-VAE-specific (FLUX); white-box fragile.

### H-D2: latent-grid-periodicity-rescale (rank 2, score 45). Explains laundering fragility and finds where traces survive downscaling
- **Mechanism:** after decoding, a resize by s puts the grid at frequency 1/(8s), which survives antialiased downscaling for s > 0.25. The JPEG block grid of the laundered image sits at k/8; s = 0.5 lands on 2/8. A cheap, training-free test of a sharp, falsifiable prediction.
- **Mandatory control:** a JPEG-history-matched arm. A real photo's own capture-JPEG grid is rescaled to the *same* period 8s.
- **Prediction:**
  - Oracle-s AUROC for SD2+SDXL vs natives: ≥ 0.80 at s = 0.6/0.7/0.8, ≤ 0.65 at s = 0.5; non-dyadic minus dyadic ≥ 0.15.
  - Blind scan: ≥ 0.80 SD-family and ≥ 0.70 all-6 at social_sim.
  - **Fail** if < 0.70 at every non-dyadic s, or the dyadic gap is < 0.05.
  - Add-on: re-score E4 pairs at jpeg75_resize07. A pair AUROC ≥ 0.70, against 0.598 at 0.5×, would mean part of E4's collapse is a dyadic artefact.
- **Weights:** none. **Compute:** < 1 h CPU, plus about 10 min GPU for the add-on [estimate].
- **Novelty:** Corvi CVPRW 2023 (arXiv 2304.06408) shows autocorrelation patterns in diffusion images. Synthbuster's detector is spectral (abstract only). Classic resampling forensics exists. No paper found predicts the grid's position after rescaling or tests the dyadic collision.

### H-D3: variance-contraction-patch-rank (rank 3, score 30)
- **Mechanism:** sampling with a denoiser and CFG contracts variability. Cozzolino et al. 2608.12155 report lower variance and effective dimensionality for regenerated images, with cues in low-to-mid frequencies, which survive laundering. INP-X Theorem 3.2 gives a VAE contraction result.
- **Prediction:** texture-residualised participation ratio ≥ 0.75 laundered, drop ≤ 0.05, FPR ≤ 2%. **Fail** < 0.65.
- **Weights:** none. **Compute:** < 30 min GPU.
- **Novelty:** multiLID (Lorenz, ICCVW 2023) is trained and on ResNet features; this is training-free, on SSL patch tokens, under laundering.
- **Main risk:** texture/complexity confound. H4's drift score failed exactly this way.

### H-D4: laundering-floor-timestep (rank 4, score 27)
- **Mechanism:** a denoiser at small t mostly sees the laundering perturbation, which is shared by both classes. At σ_t a few times larger than that perturbation, the added noise drowns it while the generator's density bias remains.
- **Pilot that runs now:** measure the laundering residual |E(L(x)) − E(x)| in SD-VAE latent space for both classes, and pre-register t* on the DDPM schedule before downloading anything.
- **Prediction:** at t*, drop ≤ 0.05 and ≥ 0.80 laundered; at t = 50, drop ≥ 0.15. **Fail** if t* < 0.70 laundered or no better than t = 50 by ≥ 0.05.
- **Weights:** SD1.5 UNet fp16, 1.72 GB, CreativeML OpenRAIL-M (sd-legacy repo; whether it is gated was not shown [unverified]). Possibly the text encoder for the empty prompt [unverified].
- **Compute:** about 130k UNet forwards, about 1.5–2.5 h [unverified throughput].
- **Novelty:** TSG, LaRE2 (t = 200), LATTE, ESIDE and DTAD pick timesteps without regard to laundering and do not test JPEG plus resize.

### H-D5: decoder-bank-universality (rank 5, score 24)
- **Pilot that runs now:** SD-VAE (GPU) and SDXL-VAE (CPU, about 11 s/pass per 009/pilot/critique.md, about 1.4 h for 440 passes). Measure cross-decoder pair AUROC and non-SD transfer.
- **Prediction:** cross-decoder ≥ 0.85 (fov512 clean); bank adds ≥ 0.03 on non-SD (SD-only: 0.875). **Fail** < 0.70.
- **Later downloads:**
  - Kandinsky 2.2 MoVQ: 271 MB, Apache-2.0, not gated.
  - FLUX VAE: 168 MB, Apache-2.0 but gated click-through, so owner approval under D-015. Oracle arm only, because FLUX is the held-out generator.

## 5. Which hypotheses could reach the Phase-5 targets?

- **AUROC ≥ 0.95 after laundering:** only **H-D1** has a plausible route. Its information ceiling on our data is the 0.98 within-pair win-rate. Whether that holds for FLUX/SD3.5 (16-channel decoders) and for DALL-E 3/Midjourney is open.
  - H-D4 could add an independent channel.
  - H-D2 predicts its own failure at exactly jpeg75_resize05.
  - H-D3 and H-D5 are supporting features or design choices.
- **FPR ≤ 1% on fresh phone photos:** H-D1 and H-D2 are scored against a real-only threshold. Docs/05 shows pooled calibration transfers across VISION devices (1.02% leave-one-device-out). Modern neural-ISP phones may look "already processed", and that risk is untested until the owner's H3 photos arrive.
- **RAID / transfer attacks:** none of these is designed for it. Reconstruction-based scores are white-box fragile (fragile-recon-arxiv2026). H-D2 is a fixed, training-free statistic and could be attacked directly. This target stays open.

## 6. Recommendation: test next

1. **H-D1 self-ref-trace-contrast:** pilot now, with no download. About 1 h GPU plus analysis. It reuses E4's cached round-trips, features and probes. It is the only candidate with a Phase-5 route, and it tests directly whether E4's within-pair signal can be used at test time.
2. **H-D2 latent-grid-periodicity-rescale:** pilot now, with no download, CPU only, so it can run in parallel with H-D1. It tells us whether the laundering loss is erasure or a dyadic collision. This changes how H-D1's L0 should be chosen, and whether D-009 should add non-dyadic resize conditions.
3. **Third pick, depending on H-D1:**
   - If H-D1 passes, run **H-D5** next to choose its decoder bank. SDXL on CPU works now; the MoVQ download is 271 MB.
   - If H-D1 fails its saturation gate, run **H-D4**: do the free laundering-floor measurement first, then ask for the UNet download.

**Downloads needed:**
- **None** for the H-D1 and H-D2 pilots.
- Later:
  - Kandinsky 2.2 MoVQ: 271 MB, Apache-2.0.
  - SD1.5 UNet fp16: 1.72 GB, OpenRAIL-M; check gating and the text encoder first.
  - FLUX VAE: 168 MB, gated, so it needs the owner. Oracle arm only.
- Total ≤ 2.2 GB, inside the ~11 GB D-015 headroom.
- The official runs still need the grip.unina.it data (RAISE, FLUX, SD3.5), which is already planned.

## 7. Web checks (2026-10-24)

Opened (WebFetch):
- https://arxiv.org/abs/2602.23732: Difference-in-Difference detection (abstract only).
- https://arxiv.org/abs/2608.12155: Cozzolino, Poggi, Verdoliva, foundation-model cues (abstract only).
- https://arxiv.org/abs/2511.20722: DinoLizer, VAE vs diffusion artefacts in inpainting localisation (abstract only). Localisation, not our setting.
- https://huggingface.co/sd-legacy/stable-diffusion-v1-5/tree/main/unet: file sizes, licence.
- https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/tree/main/movq: sizes.
- https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main/vae: size, gated.
- HF model API metadata (licence, gated flag) for FLUX.1-schnell and kandinsky-2-2-decoder: https://huggingface.co/api/models/black-forest-labs/FLUX.1-schnell and https://huggingface.co/api/models/kandinsky-community/kandinsky-2-2-decoder.

Search-result snippets only (not opened; used only to establish what exists, no numbers taken):
- arXiv 2304.06408 (Corvi, CVPRW 2023);
- arXiv 2307.02347 (multiLID, ICCVW 2023);
- arXiv 2604.14570 (ANL);
- TSG++ (MMAsia 2025, https://www.ams.giti.waseda.ac.jp/data/pdf-files/2025MMAsia_zeng.pdf);
- arXiv 2508.09487 (SARE);
- arXiv 2607.10695 (noise-residual clustering);
- arXiv 2505.14359 (DDA).

Searches that found no direct prior art:
- self-referenced feature-space VAE contrast;
- latent-grid periodicity after rescaling at an unknown scale;
- laundering-dependent timestep choice;
- training-free patch-token effective rank.

## 8. Index rows for the lead to merge into kb/index.md

| self-ref-trace-contrast | Self-referenced decoder-trace contrast under matched laundering | idea | Score how far an image moves along E4's DINOv2 trace direction when round-tripped through the SD-VAE, with the same jpeg75_resize05 applied to both sides; uses E4's 0.98 within-pair win-rate; pass = ≥0.80 laundered fov512 and ≥ pooled probe +0.10; zero downloads (N3 G4 F5 = 60) | proposed |
| latent-grid-periodicity-rescale | Latent-grid periodicity under rescaling | idea | Predicts the 8-px decoder grid survives non-dyadic resize+JPEG (≥0.80 at s=0.6-0.8) but collapses at 0.5x (≤0.65), where it lands on JPEG block harmonics; JPEG-history-matched control mandatory; CPU-only (N3 G3 F5 = 45) | proposed |
| variance-contraction-patch-rank | Variance contraction of patch-token features | idea | Texture-residualised effective rank of DINOv2 block-8 patch tokens lower for diffusion images; pass = ≥0.75 laundered with ≤0.05 drop; high texture-confound risk (N3 G2 F5 = 30) | proposed |
| laundering-floor-timestep | Laundering-floor timestep for denoiser scores | idea | SD1.5 UNet noise-prediction error at a t* pre-registered from the measured latent laundering residual: ≤0.05 laundered drop vs ≥0.15 at t≤50; needs 1.72 GB UNet (N3 G3 F3 = 27) | proposed |
| decoder-bank-universality | Decoder-bank universality of the VAE trace | idea | Leave-one-decoder-out test of a shared decoder trace (SD, SDXL, later MoVQ; FLUX VAE oracle only); pass = cross-decoder pair AUROC ≥0.85 and +0.03 on non-SD generators; does not address laundering (N2 G3 F4 = 24) | proposed |
