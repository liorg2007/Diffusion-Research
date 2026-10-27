# Attribution survey, part 3: fingerprinting REAL images (camera source ID)
Written by a survey subagent. Every number cites a kb/papers entry or a local file. [unverified] = not read from the primary paper.
Papers: kb/papers/{lukas-tifs2006, chen-tifs2008, vision-eurasip2017, noiseprint-tifs2020, densenet-cmi-arxiv2018,
sn-provenance-cnn-wifs2017, marra-gan-fingerprints-arxiv2018, spoc-arxiv2019, forensic-selfdesc-cvpr2025,
samsung-prnu-arxiv2025, sdnp-tifs2026}.md. Index rows: kb/_index_rows/camera-forensics.md.

## 1. Plain-language map
- Device level (PRNU): average noise residuals of many photos from ONE device -> reference pattern; correlate a query
  photo with it (PCE score, threshold 60 by convention). Needs pixel alignment, so full resolution and no crop/resize.
- Model level (Noiseprint, CNN model-ID): learns what a camera MODEL (demosaicing, ISP, JPEG) leaves, not the device.
  Noiseprint authors say it "cannot help for device identification" (noiseprint-tifs2020).
- Platform level: CNNs recognise WhatsApp/Facebook recompression (sn-provenance-cnn-wifs2017; numbers not read).
- Unification: Marra et al. treat GANs as cameras with PRNU-style fingerprints (marra-gan-fingerprints-arxiv2018);
  Nguyen et al. learn "forensic microstructures" from real images only (forensic-selfdesc-cvpr2025).

## 2. Reported accuracies (what we could verify)
- VISION paper's own PRNU accuracies (native vs WhatsApp vs Facebook): NOT obtained. The Springer PDF is behind a
  cookie redirect and the web tool could not read it. Do not quote any number from this doc for them [unverified].
- Independent datum (Amped blog 2020-09-18, not peer reviewed): on VISION's 35 old devices, cross-device PCE showed no
  outliers well above 60 and few weak false positives below 100. Same blog: many false positives on Galaxy S10 and later.
- DRESDEN closed-set camera-model ID: >99% on 19 models (densenet-cmi-arxiv2018, abstract via search) [unverified].
- Noiseprint: model ID 100% vs PRNU 77%; device ID 62% vs PRNU 70% on only 3 models x 2 devices (noiseprint-tifs2020).
- Modern phones: Samsung default-mode fingerprint FPR at PCE 60 is 0.95 (S9), 1.0 (S9+), 0.69 (S10), 0.66 (S10+);
  raw-mode fingerprint FPR 0 (samsung-prnu-arxiv2025, Fig. 3b). Apple portrait mode adds a shared defocus pattern that
  causes false positives (sdnp-tifs2026, abstract only).
- Generator fingerprints: PRNU-style attribution 90.3% (GANs + 2 Nikons), 90.1% after JPEG QF95; resize untested
  (marra-gan-fingerprints-arxiv2018).

## 3. How many images build a PRNU fingerprint?
- Original paper used about 320 images across nine cameras for its error-rate study [unverified, search paraphrase].
- Marra et al. use 512 residuals per source (convergence study N = 2, 8, 32, 128, 512) (marra-gan-fingerprints-arxiv2018, Sec. 3).
- Samsung paper: 20 flat-field default-mode images for a fingerprint, 40 for one device with 20 PRO-mode (samsung-prnu-arxiv2025, Sec. 3 / Table 1).
- Amped blog 2019 test used 50 reference images per phone [unverified, search snippet].
- Rule of thumb in the literature (tens of flat-field or ~50-100 natural images) is NOT something we verified from one source.

## 4. Image size requirements
- PRNU is a per-pixel pattern: query and reference must share sensor geometry. Crops and resizes need search over
  scale/shift; downscaled copies lose most of the high-frequency pattern. Social platforms resize, so classic PRNU
  on WhatsApp/Facebook copies is hard unless rescaling is estimated (general knowledge; no table read) [unverified].
- Noiseprint works on small patches (48x48 training patches) but weakens with compression/resizing (noiseprint-tifs2020).
- Apple SDNP paper: reference uses native-resolution images; 12MP 4032x3024 rear, 7MP 3088x2316 front (sdnp-tifs2026, Sec. V).

## 5. What we actually have (data/manifest_raw.csv, source_dataset == VISION, checked 2026-10-27)
- 500 native (nat), 500 natFBH (Facebook high), 500 natWA (WhatsApp) JPEGs = 1500 files; 35 devices, 14-15 native per device
  (data/raw/vision/D01..D35/{nat,natFBH,natWA}). No flat-field images, no videos.
- nat: native camera size, long side 960 (15), 2048 (14), 2560 (43), 2592 (42), 3264 (241, mostly 3264x2448), 3968 (15),
  4160 (43), 4608 (14), 4640 (30), 4784 (14), 5248 (14), 5312 (15).
- natFBH: long side 2048 (485) or 960 (15). natWA: long side 1280 (485) or 960 (15). (One device has 15 images at 960 in all three.)
- Hence the platform copies are downscaled by roughly 1.6x (FB) to 2.5x (WA) for the common 3264x2448 phones.

## 6. What is feasible for us
1. Classic device PRNU on our subset: NOT feasible. 14-15 natural images per device is below every fingerprint size above
   (20-512), natural scenes leak content, and platform copies are rescaled. Fetching more VISION nat images (flat-field set, ~4167
   flat photos per kb/datasets/vision-phone-photos.md) would fit within the 9 GB budget only if subsetted; owner approval needed.
2. Diagnostic (cheap, feasible): use VISION nat vs natWA vs natFBH as a platform-label probe. Does our detector's score separate
   WA from FB from native? If yes, social_sim is partly a platform detector (sn-provenance-cnn-wifs2017 motivates this).
3. Real-only microstructure / self-description (forensic-selfdesc-cvpr2025): re-implementable on CPU in principle; no released code found;
   would be a new baseline, not a replication. Tests on jpeg75_resize05 and social_sim are missing from the paper.
4. Noiseprint: usable only as a feature extractor; GRIP non-commercial licence, TensorFlow 1.2.1, and it was trained on VISION
   cameras, so evaluating on VISION would leak. Recommend avoiding unless the owner accepts both constraints.
5. PRNU-style generator fingerprint (Marra): cheap to run on our fakes (hundreds of same-size images per generator) as an attribution
   baseline; JPEG-resize behaviour untested and probably poor.
6. FPR risk: modern phones (Samsung, Apple portrait/HDR) inject shared patterns (samsung-prnu-arxiv2025, sdnp-tifs2026). VISION phones (2010-2015)
   cannot represent them, so any 'real fingerprint' result on VISION overstates fresh-phone performance.
7. SpoC (spoc-arxiv2019) shows camera traces can be spoofed in fakes; treat 'camera trace present' as non-proof.

## 7. Gaps
Springer-hosted papers (VISION, Amerini et al.) were unreadable; Lukas 2006 and Chen 2008 only via citations. A follow-up pass should
obtain the VISION PRNU tables and the platform-ID accuracies from an open copy (author site or the owner's institutional access).
