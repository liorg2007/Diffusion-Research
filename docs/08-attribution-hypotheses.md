# Attribution hypotheses (D-018 cycle: "can we fingerprint the source of an image?")

**Written:** 2026-10-27 by an opus/high subagent, for the lead to review and turn into the pre-registration (docs/09-attribution-plan.md). **Inputs:** CLAUDE.md, DECISIONS.md D-016…D-018, docs/03-final-report.md with both addenda, docs/07-attribution-survey-part1/2/3.md and the kb/papers entries they cite, data/manifest_raw.csv, the feature caches. **Status:** all PROPOSED; nothing was run. Numbers come from the cited file; "(derived)" = my count from data/manifest_raw.csv or a manifest; [unverified] = not checked in a primary source.

## 0. Resource facts, checked (corrections to the brief in bold)
- Fakes (data/manifest_raw.csv): Synthbuster dalle2, dalle3, firefly, midjourney-v5, stable-diffusion-2, stable-diffusion-xl, 120 each; B-Free sd3_large 150, flux 150; latent-diffusion 300 (256×256). Firefly is JPEG q≈95 under .png names (kb/datasets/synthbuster.md).
- **Caption families overlap only partly** (derived): the 8 generators span 542 RAISE-caption ids; 315 have ≥2 generators, giving 695 same-caption cross-generator pairs (10 for firefly–MJ up to 120 for dalle2–sd3). So content differs between generators unless a test is restricted to matched families.
- **Native size alone identifies many sources** (derived): dalle2, sd3 and flux are all 1024×1024. dalle3 has 3 sizes, MJ 4, firefly 4 (JPEG); SD2 and SDXL have 26 sizes each.
- **RAID/D3 is not 324 px.**
  - The 100 ids × (real + gen_0…gen_3) originals are JPEG q≈75. gen_0 is exactly 256 px, gen_1–3 are about 360–512 px, and the real short-side median is 250 px.
  - The gen_k → (SD1.4, SD2.1, SDXL, DeepFloyd IF) mapping is [unverified] (kb/datasets/raid.md; experiments/016-loso/notes.md). Ids are shared across classes (likely the same prompt [unverified]).
- **VISION:** 500 photo ids, each as an exact nat / natWA / natFBH triplet (same index across platforms; derived). 35 devices, 14–15 natives each.
  - **14 devices share 3264×2448** (D02, D04, D05, D06, D10, D11, D14, D15, D18, D19, D21, D29, D33, D34), so size cannot separate them.
  - **Same-model sets inside that group** (folder names): iPhone 4s D02/D10; 5c D05/D14/D18; 6 D06/D15; 5 D29/D34. D01/D26 (S3 Mini) are a further pair at 2560×1920. D13 (iPad 2) is 960 px in all three copies.
- **HDR+** (experiments/019-source-diversity/hdrplus_manifest.csv): 600 finals in 43 burst groups (37 hold a single model).
  - **Its labels mix marketing names and codenames:** angler = Nexus 6P, marlin = Pixel XL, sailfish = Pixel, bullhead = Nexus 5X [public codenames, unverified here]. Merged, that gives about 6 models, with the Nexus 6 at 226 images.
  - Model and resolution coincide (4208×3120 = Nexus 6, 4032×3024 = 6P/5X, 4048×3036 = Pixel/XL).
- **Cached DINOv2-base blocks 4/8/12 (CLS+mean), under 5+ conditions:**
  - experiments/004-phase4-features/cache/fov512: 2,919 rows = 8 generators + VISION 1,500 + RAISE 399 (no LDM);
  - 016-loso/cache: Open Images, plus RAID in fov224 with JPEG-only conditions; 019 cache: Unsplash, Commons, HDR+;
  - extraction speed: 2,919 images in 21 s on GPU (004/.../timings.json).
- **VAEs:** sd-vae-ft-mse runs at 0.37 s per round trip on GPU. sdxl-vae failed the GPU-vs-CPU pixel guard (LOG.md 2026-10-24 04:45), so it needs the CPU or a logit-level D-008 check. Both are MIT (env/SOURCES.md).
- **Licences and budget:** Synthbuster is CC BY-NC-SA; B-Free is GRIP non-commercial (research only). About 13.8 of 20 GB of downloads are used (env/STORAGE.md), so about 6.2 GB remain.

## 1. Sub-questions (each gets its own table; never pooled)
- **Q1 Closed-set generator attribution.** Given a fake from one of the K known generators, name it.
  - Main set: K = 8 (Synthbuster 6 + sd3 + flux). LDM (256 px) appears only in a 256-px side arm.
  - Replication set: RAID, 4-way, content-matched by id, already JPEG.
- **Q2 Open-set rejection.** Leave one generator out (LOGO). Can the model say "none of the known ones" for the held-out generator while still accepting known-generator test images?
- **Q3 Family level.** Does an image map to its decoder family (SD kl-f8 VAE / SDXL VAE / other) even for an unseen member? Example: train on Synthbuster SD2/SDXL, test on RAID SD1.4/SD2.1/SDXL.
- **Q4 Real side**, each also asked after an extra laundering:
  - (a) platform: native vs WhatsApp vs Facebook-HQ, with content held identical by the triplets;
  - (b) device: within the 14-device same-size group, and within same-model sets;
  - (c) camera model: across same-model device pairs, and in HDR+.
- **Q5 Unified source space.** One embedding where 8 generators and 8 real pipelines (VISION nat/WA/FBH, RAISE, OI, Unsplash, Commons, HDR+) are classes. Is an unseen source flagged as novel? Does its nearest known source at least have the right kind (real vs fake)?

## 2. Ranked hypotheses (one method family each)
**Common definitions.**
- **bacc** = balanced accuracy; chance = 1/K.
- **Pairwise** = on a held-out caption family that contains generators A and B: the share of (A, B) image pairs where each image scores higher for its own class than the other image does. Chance is 0.5, and the metric is free of content by construction.
- **Primary condition** = jpeg75_resize05. **Training** = T-mix (clean + jpeg75_resize05 + social_sim), as in E6. screenshot_sim and mavali_medium are held-out launderings.
- **CIs** = cluster bootstrap over families / devices / photos / groups, 1,000 replicates. **Labels** = PASS / KILL / INCONCLUSIVE, plus DISCOVERY per D-017 where noted.

| # | Method family | Main Q | Cost on 4060 Ti | Prior expectation |
|---|---|---|---|---|
| H-A1 | Foundation features by layer (DINOv2 4/8/12, cached) | Q1, Q2 | minutes, CPU | likely PASS clean; laundered open |
| H-A2 | Residual-correlation fingerprint (Marra / PRNU), training-free | Q1, Q4b | ≤1 h CPU | clean OK; laundered fail predicted |
| H-A3 | Compression-history forensics (DCT statistics) vs DINOv2 on content-identical triplets | Q4a | minutes | platform likely identifiable |
| H-A4 | Unified source space with open-set distance (Mahalanobis / kNN) | Q5, Q2 | minutes (cached) | closed-set high; novelty open |
| H-A5 | Published detectors' embeddings (CommFor, B-Free, RINE, ClipDet) | Q1 | ~1–3 h GPU [estimate] | leak-prone; open |
| H-A6 | VAE reconstruction signature (sd-vae-ft-mse, sdxl-vae) | Q3 | ~1–2 h [estimate] | laundered KILL predicted |

**H-A1. Mid-level DINOv2 features attribute generators after laundering, and not through content.**
- *Method:*
  - Logistic regression on cached fov512 block-4/8/12 features, each separately and as a concatenation. The standardiser and C (inner grouped CV) are fit on train folds only.
  - Open set: train on K−1 classes; novelty = −max softmax and the Mahalanobis distance (Ledoit-Wolf).
- *Data:* 1,020 fakes of 8 generators; 5-fold GroupKFold by caption id. Replication on RAID: 4-way, fov224 cache, jpeg(75), grouped by id.
- *Metrics:* bacc, macro-F1, confusion matrix and pairwise, per condition. LOGO open-set AUROC per generator (held-out generator vs known test images). Layer contrast b8 − b12.
- *PASS*, at jpeg75_resize05, all of: bacc ≥ 0.60 (chance 0.125) with CI lower bound ≥ texture baseline M1 + 0.10; pairwise ≥ 0.75; RAID 4-way bacc ≥ 0.55 (chance 0.25).
- *KILL:* bacc < 0.35, or pairwise < 0.60.
- *Separate label:* "open-set usable" if mean LOGO AUROC ≥ 0.80, with no generator below 0.65.
- *Anchors (not our data):* caption-matched 25-way 98.9% clean (rpa-arxiv2026 T1); ViT 95.8 → 88.67 bacc after 3 processing steps (wild-ijcnn2025 T-I); open-set AUROC 0.896 (rpa Sec. 4.4).
- *How it fails:*
  - It may attribute *style* (MJ/DALL-E 3 look) or per-generator prompt edits rather than a trace. The pairwise metric and per-pair confusions expose this.
  - n is small: about 24 test images per generator per fold.
  - Aspect-ratio crops (dalle3 1792, firefly 2688) are a residual geometry cue.
  - Firefly has a JPEG history, so every macro is also reported without Firefly.

**H-A2. A training-free residual fingerprint (average noise residual + correlation) identifies (i) generators on clean images and (ii) VISION devices, even within the same phone model.**
- *Method:*
  - Residual r = x − Wiener3×3(x) on grayscale (scipy; PyWavelets is not installed), with rows and columns zero-meaned.
  - Fingerprint = the mean train residual (generators) or the PRNU MLE Σr·x/Σx² (devices).
  - Score = normalised correlation (generators) or PCE (devices); the decision is argmax. A spectral variant (mean |FFT| of the residual) is reported but not judged.
- *(i) Generators:* top-left 256×256 crop on the native pixel grid, which keeps the 8-px VAE phase; no resize; about 70% of families for training.
  - PASS-i: clean bacc ≥ 0.60 (chance 0.125). KILL-i: clean < 0.30.
  - Laundered results use fingerprints built two ways: from clean train images (realistic) and from matched ones.
  - Prediction: social_sim < 0.40 and jpeg75_resize05 near chance. Reasons: the VAE trace fell 0.962 → about 0.60 under jpeg75_resize05 (experiments/011-vae-trace/pilot/tables.md), and Corvi's spectral peaks fade under laundering (corvi-intriguing-cvprw2023 Fig. 4).
- *(ii) Devices:* the 14 same-size VISION devices, at full resolution.
  - Fingerprints are leave-one-photo-out, from the other 13–14 natives; the test photo's WA/FB copies are excluded too. Portrait images are rotated to landscape, trying 0° and 180° and taking the max.
  - Metrics: 14-way top-1 (chance 0.071); within-model accuracy on the 4 same-model sets (chance 0.5 / 0.33); TPR/FPR at PCE 60.
  - Platform arms: WA→WA (all 1280 px), FBH→FBH, and the native fingerprint resized by 1280/3264 → WA.
  - PASS-ii: native 14-way ≥ 0.70 and within-model ≥ 0.70. KILL-ii: native 14-way < 0.30.
- *Verdict rule:* PASS if (i) or (ii) passes; both are always reported.
- *Anchors:*
  - Marra: 90.3%, and 90.1% after QF95, but with 512 residuals per source and GANs only (kb/papers/marra-gan-fingerprints-arxiv2018.md). Noiseprint device ID 62% vs PRNU 70% (noiseprint-tifs2020).
  - Literature fingerprints use 20–512 images (docs/07 part 3 §3). We have 13–14, so a fail here is weak evidence.
- *How it fails:*
  - Content can leak through edges, since one photographer shot each device's scenes. Guards: leave-photo-out, and a check that the PCE peak sits at zero shift.
  - Shared JPEG/ISP artefacts of one model can mimic a device fingerprint; that is why the within-model test is required.
  - Firefly's JPEG grid.

**H-A3. Platform history (native / WhatsApp / Facebook-HQ) is identifiable from decoded pixels alone on unseen devices, with content identical across classes, and partly survives one more laundering.**
- *Method A, compression-history features:* luminance 8×8 DCT on the native grid; normalised histograms of the first 15 zig-zag AC coefficients plus a periodicity (quantisation-step) estimate; logistic regression.
- *Method B:* cached DINOv2 fov512 blocks 4/8/12. It asks whether the "real pipeline" that the E7–E10 detectors learned includes platform processing.
- *Data:* 485 triplets (D13 excluded: its 15 copies are never resized); 5-fold GroupKFold by device, so all copies of a photo stay together.
- *Metrics:* 3-way bacc (chance 0.333), confusion matrix, per device. Conditions: clean, plus social_sim and jpeg75_resize05 applied on top. Reference M0 = file size + quantisation-table hash: the trivial bar, expected at about 1.0.
- *PASS:* A or B reaches ≥ 0.80 clean and ≥ 0.60 at social_sim. *KILL:* A < 0.60 and B < 0.50, clean.
- *Exploratory camera-model arm (no label):*
  - VISION: train on one device of a same-model pair, test on the other, against the other models (DINOv2 block 4, fov512).
  - HDR+: model ID grouped by burst group, with codenames merged.
- *How it fails:*
  - At native resolution, method A's block statistics also read the output scale (the platform's resize). Method B (fov512) equalises scale.
  - VISION phones are 2010s devices; modern phone pipelines are untested (03-final-report §8).

**H-A4. In one DINOv2 source space, an unseen source (generator or real pipeline) is flagged as novel, and its nearest known source has the right kind.**
- *Method:* 16 classes (8 generators + VISION nat/WA/FBH, RAISE, OI, Unsplash, Commons, HDR+) from caches 004/016/019. The cache specs must be checked equal, and D-008 must pass, first.
- *Leave-one-source-out:*
  - novelty = Mahalanobis or kNN-10 distance, scored as AUROC of held-out-source images vs known-source test images;
  - "kind accuracy" = share of held-out images whose nearest known source is of the same kind, compared with the E10 k = 5 per-source FPR (019/results/tables.md).
- *PASS:* mean novelty AUROC ≥ 0.85 at jpeg75_resize05, with min ≥ 0.70. *KILL:* mean < 0.70.
- *DISCOVERY tier:* kind accuracy ≥ 0.95 (FPR ≤ 5%) for every held-out *real* source (D-017).
- *Prior:* real sources differ from each other almost as much as from fakes (C2ST 0.977 vs 0.984, 03-final-report addendum E8). So closed-set source ID should be high; novelty is the open question.
- *How it fails:* novelty may track the content domain (OI vs Commons) rather than the pipeline. All fakes carry RAISE-caption content, so a held-out generator sits near RAISE.

**H-A5. Embeddings of detectors trained with augmentation on many generators attribute better under laundering than DINOv2.**
- *Method:* penultimate embeddings via forward hooks: CommFor-384 ViT, B-Free DINOv2-reg4, RINE (CLIP-L/14 per-block CLS stack), ClipDet (CLIP-L/14). Each gets the H-A1 probe and splits.
- *PASS:* best embedding bacc at jpeg75_resize05 ≥ the H-A1 best layer + 0.05, with paired family-bootstrap CI lower bound > 0. *KILL:* ≤ the H-A1 best layer.
- *Leaks, reported per class:* CommFor was trained on VISION reals and 4,763 HF models, mostly LDM (kb/papers/communityforensics-cvpr2025.md); B-Free on SD2.1 inpaintings (kb/papers/bfree-cvpr2025.md); ClipDet on LDM (kb/papers/clipdet-cvprw2024.md).
- *How it fails:* binary real/fake training may collapse generator identity into one "fake" direction.

**H-A6. A two-VAE reconstruction signature identifies the decoder family (SD kl-f8 / SDXL / other), and the signature survives jpeg75_resize05.**
- *Method:* about 8 features from native 512 crops. Per VAE: L2 error of one round trip and of a double round trip, the AEDR ratio, and a high-pass error by 8-px phase. Read-outs: an argmin rule (no training) and logistic regression.
- *Data:* train on Synthbuster (SD2 → kl-f8, SDXL → XL, the other 6 → other). Test cross-dataset on RAID originals (already JPEG q≈75) once the gen_k mapping is verified.
- *PASS:* 3-way bacc ≥ 0.70 clean and ≥ 0.50 at jpeg75_resize05 (chance 0.333). *KILL:* < 0.45 at jpeg75_resize05.
- *Prediction: KILL.*
  - DNA end-to-end falls to 9.15% at QF75 (kb/papers/dna-arxiv2026.md); our E4 trace fell 0.96 → 0.60; reconstruction scores failed three times on laundered inputs (03-final-report §4.2).
  - sd-vae-ft-mse may not be exactly SD2's VAE [unverified], and we have no SD3/FLUX 16-channel VAE.

## 3. Validity gates (mandatory for every hypothesis; each run's notes.md must show them ticked)
1. **Identical preprocessing.**
   - Decode, strip EXIF, convert to RGB; read the format from content (Firefly).
   - Feature methods use fov512 for every class. Pixel-grid methods (H-A2, H-A6) crop on the native grid only, with no resize in "clean".
   - Laundering uses experiments/lib/perturb.py through one code path for all classes. Nothing may depend on label or source (lesson of E9).
2. **Baselines that must be beaten** (always shown):
   - **M0 metadata:** width, height, aspect, true format, bytes/pixel and quantisation-table hash, on raw files. It is the trivial reference and shows how much "attribution" comes for free.
   - **M1 texture:** about 10 low-level statistics on the *preprocessed* tensor (Laplacian variance, 8×8 blockiness, noise std, saturation, colour moments). Laplacian alone separated real vs fake at AUROC 0.778 (03-final-report §4.1).
   - A method counts only if its CI lower bound beats M1 by ≥ 0.10 bacc, and M0 on preprocessed tensors is at chance.
3. **Grouped splits.**
   - By caption id: all generators and the RAISE twin move together. RAID by id. VISION by device (platform, model) or by photo triplet (device ID). HDR+ by burst group.
   - Pairwise scoring on matched families is mandatory for Q1. Folds and seeds are committed before any feature is read.
4. **Laundering.**
   - Conditions: clean, jpeg75_resize05 and social_sim (D-009), plus screenshot_sim and mavali_medium, which are never seen in training.
   - Report both clean-train transfer and T-mix.
   - Thresholds and novelty cut-offs are refit per condition only if pre-registered, because they do not transfer (03-final-report §4.5).
5. **Label-leak checklist.**
   - No test family, device, photo or burst appears in training. Calibration and novelty statistics use train-role data only. Copies of one photo stay together.
   - No layer or hyper-parameter is chosen on test folds: use inner CV, or report all with Holm. Detector training-set overlap is flagged per class.
   - A 200-shuffle permutation control must give chance. Dedupe by sha256 across classes. Run the D-008 GPU vs CPU guard on any new GPU pass.
6. **Chance levels in every table:**
   - bacc 1/K: 8-way 0.125, RAID 0.25, family 0.333, platform 0.333, 14 devices 0.071, same-model 0.5 / 0.33;
   - pairwise and AUROC 0.5; plus the majority-class rate.

## 4. Recommended order, stop-rule budget, downloads
**Order** (cheapest and most informative first): M0 + M1 baselines → H-A1 (cached) → H-A3 (cached, plus a cheap DCT pass) → H-A2 (generators first, then devices) → H-A4 (reuses H-A1) → H-A5 → H-A6.

**Stop-rule budget.** D-018 stops the cycle after 3 failed pre-registered hypotheses, and H-A6 (and the laundered part of H-A2) are predicted negatives. I suggest the lead pre-registers H-A6 as a non-counting *diagnostic*, or runs it only if fewer than 2 hypotheses have failed. That decision goes in docs/09.

**Downloads.** About 6.2 GB is left. Synthbuster is NC, so the owner must approve per CLAUDE.md. Sizes are from kb/datasets/*.md.
- **Tier 1, recommended (about 0.4 GB):**
  - Synthbuster SD1.3, SD1.4 and GLIDE, 300 each by zip range-fetch: about 0.16 + 0.16 + 0.03 GB (folder sizes 0.54 / 0.54 / 0.09 GB per 1,000). This adds a near-twin test (SD1.3 vs SD1.4, assumed to share architecture and VAE [unverified]) and a shared-VAE-family test (SD1.x vs SD2 [unverified]) for Q2 and Q3.
  - RAID: 400 more ids of originals (real + 4 generators), at about 182 KB per id (derived), so about 0.07 GB. MIT card; upstream D3 licence [unverified]. This makes the 4-way content-matched replication 5× larger.
- **Tier 2, only if the H-A1/H-A2 CIs are limited by n:**
  - Synthbuster +180 per existing generator (to 300 each): about 2.0 GB (0.18 × 11.2 GB of folders).
  - B-Free sd3/flux +150 each: about 0.5 GB (sd3_flux.zip is 3.4 GB for 2,000).
- **Tier 2, only if H-A2(ii) is INCONCLUSIVE:** VISION flat-field images, 30 per device for the 14-device group. About 1.3 GB at about 3 MB per image [estimate]; CC BY-SA.
- **Not worth it now:** WILD/DRAGON (licence or size unknown, docs/07 part 1 §5); SD/FLUX U-Nets (DNA/SDS stage 2 need GBs each, docs/07 part 2 §4).
