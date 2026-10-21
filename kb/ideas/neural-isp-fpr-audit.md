# Do computational-photography modes (night, portrait, super-res zoom) on modern phones break real-photo FPR?
- **id:** neural-isp-fpr-audit
- **Gap addressed:** G3 (no FPR data on 2021+ neural-ISP phones), G1 (threshold transfer)
- **Scores (1-5):** novelty 4 · expected gain on unseen/in-the-wild 3 (diagnostic, but on the exact target metric) · Mac feasibility 3 (needs ~250 owner photos) · **product 36** (rank 3)

- **Hypothesis:** At a threshold calibrated on VISION phone photos (2015-2017 devices) for 1% FPR, fresh photos from a 2021+ smartphone have FPR >= 2x the VISION held-out-device FPR for at least 2 of 4 detectors, and photos taken in computational modes (night, portrait/bokeh, >= 5x digital/super-resolution zoom) have FPR >= 2x that of the same phone's default mode.

- **Rationale:**
  - The project's validation target is FPR <= 1% on fresh phone photos, yet the only large phone study (laion-mobile-arxiv2026) has only 212 images (0.03%) from phones released 2021+, with the pool peaking 2014-2016; its era analysis (Sec 5.4) found no significant FPR rise for heavier-ISP phones, but it is underpowered for modern devices and does not split by capture mode. VISION is also 2015-2017 era (kb/datasets/vision-phone-photos.md).
  - Multi-frame fusion, learned denoising, synthetic bokeh and learned super-resolution produce exactly the smooth, low-noise, upsampled textures that low-level detectors key on (up-sampling traces: npr-cvpr2024; spectral peaks: corvi-icassp2023). If these modes raise FPR, VISION-based calibration (worst-env-conformal-fpr) is insufficient and modern-phone data must enter calibration.
  - Result is useful either way: refutation means VISION-based calibration transfers to modern phones (good news for the cheap path).

- **Closest prior work:**
  - KB: laion-mobile-arxiv2026 (12 detectors on ~9k EXIF-identified phone photos; FPR 17-91% at modern-content EER threshold; era effect not supported), truefake-ijcnn2025 (platform sharing raises FPR for some detectors, e.g. MISLNet TNR 0.41 on FFHQ via Facebook, Table IV), vfm-baseline-arxiv2025 (VLM backbones flag many reals: CLIP 46.7, PE 44.4 real accuracy on WebAIG-25, Table 4).
  - Web check: search found no study measuring classifier FPR by capture mode on modern phones. Closest: LAION-Mobile itself (https://arxiv.org/pdf/2609.11134); a Forensic Science International 2026 paper on PRNU camera-fingerprint false matches for AI images (https://dfrws.org/wp-content/uploads/2026/03/Boon-or-Bane-Source-Camera-Identificati_2026_Forensic-Science-International.pdf - camera attribution, not AI-image classifiers); vendor/press anecdotes about low-light phone shots being flagged (not evidence). The HF card https://huggingface.co/GPTchatly/slope-scope states "heavily retouched real portraits are often called AI" (non-peer-reviewed).

- **Evidence for / against:**
  - For: anecdotal and vendor reports (above); TrueFake and LAION-Mobile show real-side FPR is fragile to processing/threshold.
  - Against: LAION-Mobile Sec 5.4 - predicted FPR rise with heavier ISP not supported; four detectors moved the opposite way.

- **Test plan (Mac-feasible):**
  - Data (owner action required): ~250 fresh photos from the owner's own 2021+ phone, stored as JPEG (iOS "Most Compatible" or Android default), labeled by mode: default ~100, night ~50, portrait ~50, >= 5x zoom ~50; plus WhatsApp-sent copies of all. Photos stay local, are never committed to git or published; only scores and EXIF model/mode are logged.
  - Detectors: B0 frozen-DINOv3 probe, UnivFD (MIT), CommFor ViT-S (MIT), B-Free (research-only weights). Thresholds: M3 from worst-env-conformal-fpr fitted on VISION D_eval at alpha = 1%; also M2.
  - Metric: FPR with Clopper-Pearson 95% CI, per mode and pooled; compare with VISION leave-device-out FPR.
  - **Supported if:** pooled modern-phone FPR >= 2x VISION held-out FPR (and CIs do not overlap) for >= 2 of 4 detectors, AND computational-mode FPR >= 2x default-mode FPR for >= 2 detectors. **Refuted if:** modern-phone FPR lies inside the VISION FPR 95% CI for all 4 detectors. Note: n=100 default photos gives a CP upper bound of 3.6% even at 0 errors, so this audit can detect large effects only; pre-registered as a screen, not a full-scale validation.
  - Cost: < 10 min of inference after feature caching.

- **Status:** proposed
