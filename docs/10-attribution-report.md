# Attribution report: can we fingerprint the source of an image? (D-018 cycle)

**Date:** 2026-10-28. **Author:** fable/high report agent, for the lead and the owner. **Status:** final report of the D-018 cycle. The stop rule has been reached (D-019), so nothing below the line "next steps" may start without an owner limit raise.
**Grounding.** Every number cites the file it comes from. Short paths: `020/` = `experiments/020-attr-foundation/`, `021/` = `experiments/021-attr-residual/`, `022/` = `experiments/022-attr-platform/`, `023/` = `experiments/023-attr-detector-emb/`, `024/` = `experiments/024-attr-vae/`. Where an opus/high critic corrected a run's number or reading, **the critic's version is used** and the difference is stated. "(derived)" = arithmetic done for this report; [unverified] = not checked in a primary source.

---

## 1. The answer in one paragraph

**Partly, and mainly for sources we already know.** For AI images, a simple linear classifier on the *middle* layers of a frozen DINOv2 network names which of 8 known generators made an image 85.5% of the time even after JPEG-75 plus half-size resizing (chance 12.5%), and 81–87% under launderings it never saw in training. That only works if it is trained on laundered copies: trained on clean images alone it falls to 54% (`020/results/tables.md`). Three things do not work. (1) Near-identical versions (SD1.3 vs SD1.4) cannot be told apart (chance) (`020/secondary/tables.md`). (2) The method cannot reliably say "this comes from a generator I don't know" (mean AUROC 0.77, below the 0.80 bar) (`020/results/tables.md`). (3) Low-level fingerprints are real on untouched files but do not survive laundering: the VAE-decoder signature is 0.97 on clean files and at chance after one JPEG+resize (`024/results/tables.md`). For real photos, classic sensor-noise fingerprinting (PRNU) identifies which of 14 same-size 2010s phones took a photo 98.5% of the time. It still works on WhatsApp (93%) and Facebook (87%) copies (`021/results/tables.md`). That is a textbook replication, and it is untested on modern phones. Which platform a photo passed through (native / WhatsApp / Facebook) is readable from file metadata, but not from laundered pixels (`022/results/critique.md`). Pre-registered score: 2 PASS (H-A1; H-A2 via its device arm only), 3 counted failures (H-A3, H-A4, H-A5), so the D-018 stop rule is reached (D-019).

## 2. Glossary (each term explained once)

| Term | Meaning |
|---|---|
| **Attribution** | Naming the source of an image: which generator, or which camera/device/platform. Detection (the earlier cycle, `docs/03-final-report.md`) only asks "real or fake?". |
| **Closed set / open set** | Closed set: the image is assumed to come from one of the K known sources, so pick one. Open set: the answer may also be "none of these, unknown source". |
| **bacc** | Balanced accuracy: the average over classes of "share of that class named correctly". It is not fooled by unequal class sizes. |
| **Chance** | What guessing gives: 1/K for bacc (8 generators 0.125, 3 platforms 0.333, 14 devices 0.071); 0.5 for AUROC and pairwise. |
| **Pairwise (2-AFC)** | Take two images made from the *same caption* by generators A and B, and ask whether the model ranks each one as more like its own generator. Content is identical within a pair, so content cannot help. Chance 0.5. |
| **LOGO / novelty AUROC** | Leave-one-generator-out: train without generator X, then measure how well a "novelty score" separates X's images from known-generator test images (AUROC; 0.5 = no separation). "Open-set usable" was pre-defined as mean ≥ 0.80 with no generator below 0.65. |
| **PRNU / PCE** | PRNU: each camera sensor has tiny, fixed pixel-to-pixel sensitivity differences, a "sensor noise fingerprint". You average the noise of a device's photos into a reference pattern, then correlate it with a new photo. PCE is the correlation-peak score; ≥ 60 is the conventional "match" threshold. |
| **Laundering conditions** | Processing applied identically to all classes (`experiments/lib/perturb.py`): `clean` (none); `jpeg75_resize05` (half-size, then JPEG quality 75, the **primary** test); `social_sim` (random 0.7–1.0 resize + JPEG 70–100). `screenshot_sim` and `mavali_medium` were **never used for training** (held-out launderings). |
| **T-mix vs clean-train** | T-mix trains on clean + jpeg75_resize05 + social_sim copies. Clean-train trains on clean images only and tests on laundered ones (the "realistic surprise" case). |
| **M0 / M1** | The trivial baselines every method had to beat (docs/08 §3). **M0** = file metadata only (width, height, format, bytes per pixel, JPEG quantisation table). **M1** = about 10 simple texture statistics (blur, blockiness, noise, colour). A method counted only if its CI lower bound beat M1 by ≥ 0.10. |
| **b4 / b8 / b12 / cat** | DINOv2-base block 4, 8 or 12 features (12 = last, most "semantic"), or all three concatenated. |
| **CI** | 95% cluster-bootstrap interval (resampling caption families, devices or photos), shown as [low, high]. |
| **PASS / KILL / INCONCLUSIVE, "failure"** | Pre-registered labels (docs/08, docs/09). A counting hypothesis "fails" if KILL, or INCONCLUSIVE with the estimate on the KILL side. 3 failures stop the cycle (D-018). |
| **D-008 guard** | GPU features must match CPU features (max logit difference ≤ 1e-3) on a fixed subset, or the run is invalid. |

---

## 3. Data and validity gates

**Data** (docs/08 §0, `020/secondary/notes.md`, `021/notes.md`, `022/notes.md`):
- **Main generator set:** 1,020 fakes from 8 generators (Synthbuster DALL-E 2, DALL-E 3, Firefly, Midjourney v5, SD2, SDXL at 120 each; B-Free SD3 and FLUX at 150 each), made from 542 RAISE caption families.
- **Secondary:** Synthbuster SD1.3, SD1.4 and GLIDE (300 each); RAID 4-way set, 100 ids first, then 500 (2,000 fakes).
- **Real side:** VISION, 485 content-identical native/WhatsApp/Facebook triplets from 34 devices; 14 same-size (3264×2448) devices with 197 photos for PRNU; HDR+ 600 phone finals.
- **Unified 16-source space:** 5,313 images (8 generators + 8 real pipelines).

**Gates that made the results trustworthy** (docs/08 §3, ticked in each notes.md):
1. **Identical processing** for every class, through one code path, with EXIF stripped.
2. **Splits grouped** by caption family, device, photo or burst, so the same content never sits in both train and test. Folds were committed before features were read (`020/results/folds_commit.json`).
3. **M0 and M1 shown in every table.** M0 alone already names the generator at 0.688 from raw files (`020/results/tables.md`). That number shows how much "attribution" is free from metadata, and why every method works on decoded, preprocessed pixels.
4. **Permutation controls at chance** in every run, plus sha256 dedupe. Dedupe caught 6 pixel-identical iPad WhatsApp/native copies, which were dropped before folds were made (`020/notes.md`).
5. **An independent opus/high critic per result**, who reran checks in each `critic/` folder.

**Gate failures that changed verdicts:**
- **H-A3, method A (platform from DCT statistics).** The run reported clean 0.929 and "not a failure". The critic found that image *size* alone gives platform at **0.990 [0.971, 1.000]** on A's input (WhatsApp is always 1280 px, Facebook 2048 px). A was in effect reading the last JPEG quantisation grid, i.e. metadata recovered from pixels. Gate 2 fails for A, so the A arm is INVALID. Rerun at a common scale, A gives **0.690 clean and 0.365 / 0.366 laundered** (chance 0.333). That moved H-A3 from "not a failure" to **counted failure 3 of 3** (`022/results/critique.md` §1, §3; D-019).
- **D-008 in H-A6 (VAE).** sdxl-vae GPU features failed the guard (logit difference 0.15–0.23 vs 1e-3), and sd-vae GPU features failed marginally. The first analysis was therefore INVALID (kept as `024/results/results_before_cpu_sdvae_D008fail.json`). The reported run uses CPU features only, on a capped subset of 360 images (`024/notes.md`). The critic then found that two **CPU** runs differ by 1.1e-2, as much as GPU vs CPU. The 1e-3 guard measures the VAE round trip's own instability, not GPU error (`024/results/critique.md` §4). The cap was probably unnecessary.

**Gates found weak (no label change):**
- "M0 on preprocessed tensors at chance" is true by construction (every tensor is 512²), so it cannot catch resampling history. The critic replaced it with size-matched checks (`020/results/critique.md` §1).
- The pairwise criterion was saturated: unmatched-content pairs already give 0.991 (`020/results/critique.md` §2).

---

## 4. Results per hypothesis

| # | Method | Label | Counts toward the stop rule? | Key number (primary condition) | Critic |
|---|---|---|---|---|---|
| H-A1 | Mid-level DINOv2 probe, 8 generators | **PASS** | no | 0.855 [0.834, 0.876] | confirmed |
| H-A1 side label | Open-set usable | **NO** | (not a hypothesis) | mean LOGO AUROC 0.768 | confirmed |
| H-A2 | Residual / PRNU fingerprint | **PASS via (ii) devices only**; (i) generators INCONCLUSIVE | no | devices 0.985; generators 0.522 (ties M1) | confirmed; "(i) uninformative" |
| H-A3 | Platform history (native/WA/FB) | **INCONCLUSIVE, counted as failure** | **failure 3** | gate-valid: 0.690 clean, 0.365 laundered | A arm INVALID (D-019) |
| H-A4 | Unified 16-source space + novelty | **KILL**; DISCOVERY NO | **failure 2** | novelty AUROC 0.531 | confirmed, robust |
| H-A5 | Published-detector embeddings | **KILL** | **failure 1** | best 0.740 vs DINOv2 0.858 | confirmed; RINE deviation logged |
| H-A6 | Two-VAE reconstruction signature | INCONCLUSIVE (non-counting diagnostic) | n/a | clean 0.967; laundered 0.333 | "clean-only PASS, laundered KILL" |

**Stop-rule count: 3 of 3 (H-A5, H-A4, H-A3)** → D-018 stop rule reached (D-019). All five counting hypotheses had already run.

**H-A1 — DINOv2 middle layers attribute generators after laundering.** Sources: `020/results/tables.md`, `020/results/critique.md`, `020/secondary/tables.md`.
- **Headline:** at jpeg75_resize05 with T-mix, 8-way bacc 0.855 [0.834, 0.876]. M1 gets 0.425 (paired gain +0.430 [+0.398, +0.464]).
- **Other conditions:** clean 0.913, social_sim 0.872. Never-seen launderings: screenshot_sim 0.866 and mavali_medium 0.813.
- **Clean-train transfer collapses:** 0.542.
- **Layers:** b12 (last) gives only 0.562. The b8 − b12 contrast is **+0.254 [+0.221, +0.287]**.
- **Not a size shortcut** (critic). Generators that are all 1024² (DALL-E 2 / SD3 / FLUX) give a 3-way **0.964 [0.943, 0.981]**. SD2 vs SDXL, which share identical size sets, gives 0.933 [0.901, 0.963].
- **Pairwise:** the run reports 1.000, but the critic showed it is saturated (unmatched 0.991; strict reading 0.988). Read it as "no reliance on caption content", not as extra evidence.
- **RAID replication:** 0.688 [0.650, 0.725] at n = 100 ids; **0.772 [0.756, 0.789] at n = 500**. Fresh ids alone give 0.770. Without the size-cued gen_0 class, a 3-way probe gives 0.736 [0.715, 0.756] (chance 0.333). Clean-train again collapses (0.497).
- **Weakest class:** Midjourney v5 (recall 0.667).
- **Near-twins (secondary arm):** SD1.3 vs SD1.4 best cell 0.527 [0.490, 0.562], inside the label-swap null (p95 0.54).
- **11-way closed set** (adding SD1.3, SD1.4, GLIDE): 0.765 [0.747, 0.783]. SD1.x forms one cluster split 50/50 between its two versions.
- **Unseen-generator behaviour:** SD1.x images fed to the 8-way probe spread over SD2, MJ and SD3. GLIDE is called DALL-E 2 91% of the time, but that is confounded by GLIDE's 2× upscale.
- *Meaning:* known generators *as deployed* can be named after laundering. The critic reads the cue as **mid-level "rendering style"**, not semantics and not a pixel fingerprint, so it is likely version-specific.

**H-A2 — training-free residual fingerprints.** Sources: `021/results/tables.md`, `021/results/critique.md`.
- **(ii) Devices, native photos:** 14-way top-1 0.985 [0.969, 1.000]; within same phone model 1.000; TPR at PCE 60 0.964.
- **Through platforms:** WhatsApp→WhatsApp 0.930, Facebook→Facebook 0.871, native fingerprint→WhatsApp 0.915. TPR at PCE 60 drops to 0.37–0.45, because argmax needs no threshold but PCE 60 does.
- **False-positive rate.** The run reports FPR 0 [0, 0]. The critic corrects this: 2,561 false pairs per arm give a rule-of-three upper bound of **≤ 0.0012**, and the largest false PCE was 49.9 (native) / 55.7 (native→WA), just under 60. So "FPR 0" is fragile.
- **Baselines:** device M0 (metadata) 0.498. Device M1 was missing from the run; the critic computed **0.346**.
- **Content controls (critic):** a near-duplicate probe found that content does not drive PCE.
- **(i) Generators:** clean 0.522 [0.478, 0.567] vs M1 0.530, and raw-metadata M0 is better (0.674). With clean-built fingerprints, laundered images score 0.114 / 0.164 (chance).
- **The "unexplained" 0.523** (matched laundered fingerprints) is explained by the critic: the signal is locked to the image's top-left corner. A (4,4)-pixel shift drops it to 0.125, and on clean images a (3,5) shift gives 0.153. It is an oracle result.
- *Meaning:* PRNU is a replication of Lukáš 2006 / Chen 2008. Notable only because 13–14 natural photos per device suffice, against the 20–512 images in the literature. That contradicts `docs/07-attribution-survey-part3.md` §6.1 ("NOT feasible").

**H-A3 — platform history from pixels.** Sources: `022/results/tables.md`, `022/results/critique.md`.
- **Run numbers.** Method A: 0.929 / 0.523 / 0.584 (clean / social_sim / jpeg75_resize05). Method B (DINOv2 b8): 0.610 / 0.425 / 0.399. M0 from headers: 0.985.
- **Critic-corrected:** A is INVALID (size 0.990, see §3). A at a common scale gives 0.690 / 0.365 / 0.366. M1 on A's native-grid input reaches 0.619 clean, i.e. M1 itself reads scale.
- **Exploratory, no label:**
  - VISION camera model, from one device to its same-model twin: AUROC 0.747 [0.679, 0.810].
  - HDR+ 6-model ID: 0.200 (chance 0.167; permutation max 0.188), i.e. no usable signal.
- *Meaning:* platform history lives in file headers and in the untouched JPEG grid. After one more laundering it is gone. The detectors' DINOv2 features carry little platform information (0.61 clean, about 0.40 laundered). This agrees with the platform-flat FPR in E10: native 0.011, WhatsApp 0.011, Facebook 0.008 (`022/results/critique.md` §6).

**H-A4 — one source space that flags unseen sources.** Sources: `020/results/tables.md`, `020/results/critique.md`.
- Mean novelty AUROC 0.531 (min 0.386, HDR+). Removing the VISION sibling pools leaves the 8 fakes at 0.504 (critic). Every layer and score variant falls in 0.51–0.55.
- DISCOVERY NO: the minimum "nearest known source has the right kind" accuracy is 0.733 (Unsplash).
- The closed set itself works: 16-way 0.695, with real/fake kind accuracy 0.952. Raw metadata M0 gives 0.832 on the same 16-way task.
- *Meaning:* naming a known source works; noticing an *unknown* one does not.

**H-A5 — embeddings from published detectors.** Sources: `023/results/tables.md`, `023/results/critique.md`.
- At jpeg75_resize05: CommFor 0.740, ClipDet 0.709, B-Free 0.660, RINE 0.157 (collapsed, effective rank 2).
- Best detector minus DINOv2 cat: **−0.118 [−0.149, −0.087]**. Every detector is below cat in all 5 conditions. Open-set mean AUROC is ≤ 0.656.
- **Critic notes:**
  - On clean images CommFor ≈ b4 (−0.016); the gap opens under laundering.
  - Detectors were probed only at their last layer.
  - The RINE probe deviated from the pre-registration (proj2 output instead of the per-block CLS stack). This does not affect the label, because CommFor and B-Free test the mechanism.
  - Fusion cat+CommFor gives 0.880 vs 0.861 (oracle C, descriptive only).
- *Meaning:* real/fake training squeezes generator identity out of a detector's final embedding. A generic backbone's middle layers keep more of it.

**H-A6 — VAE decoder signature (non-counting).** Sources: `024/results/tables.md`, `024/results/critique.md`, `024/notes.md`.
- **Clean:** 3-way family bacc **0.967 [0.933, 0.993]**, +0.214 over M1. A no-training argmin rule separates SD2 from SDXL at 1.000; size buckets are matched (critic).
- **Laundered:** clean-trained 0.333 (everything is called "other"). Matched training gives 0.584 / 0.591, below M1 (0.735 / 0.701). The argmin rule falls to 0.453 / 0.484.
- **What it identifies** (critic):
  - **decoder weights, not families**: SD1.x falls off the sd-vae-ft-mse manifold (0% kl-f8 on clean, border effect ruled out);
  - all 16 Midjourney v5 images prefer sd-vae-ft-mse; without MJ, clean bacc is 0.991.
- **Small n:** 108 test images, 4–10 per "other" generator.

---

## 5. What worked, what didn't, and why

**The pattern.** An image carries source traces at three depths. Laundering removes them from the bottom up.
1. **Pixel-level traces exist on pristine files.**
   - VAE decoder match: 0.967; SD2 vs SDXL 1.000 with no training (`024/`).
   - Generator residuals carry *something* (0.522 clean), but it is locked to the image corner (`021/results/critique.md` §3).
   - File metadata alone: 0.688 for generators, 0.985 for platforms (`020/`, `022/`).
2. **One JPEG-75 plus a half-size resize erases them.** The VAE signature drops to 0.333 and generator residuals to 0.114.
   - This repeats the detection cycle's finding (VAE trace 0.962 → ≈ 0.60, `docs/03-final-report.md` §4.2) and the literature (DNA 9.15% end-to-end at QF75, `kb/papers/dna-arxiv2026.md`).
   - A 0.5× downscale throws away 3 of every 4 pixels, and JPEG quantises what is left. The pattern these methods read is exactly what goes.
3. **Learned mid-level features survive, if they are trained on laundered data.**
   - DINOv2 b4/b8 give 0.855 at jpeg75_resize05. They transfer to launderings never seen in training at 0.866 (screenshot) and 0.813 (mavali).
   - Without laundered training they drop to 0.542, so the robust cues must be *taught*. Training on laundered data teaches cues that survive a family of launderings, not just the one used.
   - The last, semantic layer is worst (0.562). The useful information is "how the generator renders" (texture, tone, smoothness), not "what is in the picture".
   - Real/fake detectors are worse at this than a generic backbone (H-A5, −0.118).

**Where the limits are.**
- **Near-twin versions are inseparable.** SD1.3 vs SD1.4 sits at chance in every feature set and condition (`020/secondary/tables.md`). If two versions render alike, a style cue has nothing to grip.
- **Open-set and novelty fail.**
  - A closed-set probe must pick one of its known classes. Its confidence separates unknown generators only weakly (0.768), and distance-based scores are at chance (Mahalanobis 0.546; 16-source novelty 0.531).
  - Unknown generators are silently mapped to known ones: SD1.x → SD2/MJ/SD3; GLIDE → DALL-E 2.
  - The literature also reports open-set as the hard part (POSE near-twin AUC 68.15; WILD open-set ROC-AUC 0.801, `docs/07-attribution-survey-part1.md` §1).
- **Device PRNU is the exception on the real side.** It survives WhatsApp/Facebook (0.930 / 0.871 top-1). It has the conditions the generator traces lack:
  - a reference fingerprint from the *same* device;
  - millions of pixels at full resolution;
  - a pattern fixed in the sensor rather than in a decoder that laundering rescales.
  - It is a replication, on 2010s phones only.
- **Platform history does not survive in pixels.** Once the original JPEG grid is broken, native, WhatsApp and Facebook copies are indistinguishable (0.365) (`022/results/critique.md` §4).

---

## 6. Honest novelty assessment

Compared only against the KB papers cited in the critiques. This was not a full literature search, so "plausibly new" means "not reported in the ~30 attribution papers we read".

**Replications (expected results, now confirmed on our data):**
- PRNU device ID within the same phone model (`kb/papers/lukas-tifs2006.md`, `chen-tifs2008.md`, `vision-eurasip2017.md`).
- Closed-set attribution is high on clean data. Anchors: 98.9% 25-way caption-matched (`kb/papers/rpa-arxiv2026.md`); ViT 95.8 → 88.67 after 3 processing steps (`kb/papers/wild-ijcnn2025.md`).
  - Our 0.855 laundered is a solid baseline, **not state of the art**, and not like-for-like (`020/results/critique.md` §5).
- Reconstruction/VAE traces die under JPEG (`kb/papers/dna-arxiv2026.md`).
- Open-set is hard (`kb/papers/pose-cvpr2023.md`, `wild-ijcnn2025.md`).
- Intermediate layers beat the last layer. RINE shows +9 ACC points for *detection* (`kb/papers/rine-eccv2024.md`, Table 4).

**Plausibly new or useful (small n, unreviewed):**
1. **The middle-layer advantage measured for *attribution under laundering*:** b8 − b12 = +0.254 [+0.221, +0.287] (+0.127 with clean training).
2. **A laundering-trained probe transfers to unseen launderings** (0.866 / 0.813), with the clean-train collapse quantified (clean-trained: 0.942 on clean → 0.542 at jpeg75_resize05).
3. **Size-controlled checks:** ≥ 0.92 within identical-size generator blocks. The critic notes most attribution papers lack these.
4. **Negative results, each with its control:**
   - near-twins at chance against a label-swap null;
   - unified-space novelty at chance (0.531);
   - detector embeddings worse than generic DINOv2 for attribution (−0.118);
   - VAE signatures identify decoder weights, not families. SD1.x is off the sd-vae-ft-mse manifold, and MJ v5 is on it (an inference; MJ's decoder is [unverified]).
5. **Platform information is absent from detector features after laundering** (0.40 vs chance 0.33), consistent with the platform-flat FPR in E10.
6. **PRNU from 13–14 natural photos per device** (2010s iPhones, 8 MP).

---

## 7. Practical implications for a detection system

**A suite trained on known generators gets attribution almost for free as a side output.** It uses the same DINOv2 block-8 features and laundering-augmented training (T-aug) that the detection report recommends (`docs/03-final-report.md` §5.1). Retraining is cheap: features for 2,919 images took 21 s on GPU (docs/08 §0), and a linear probe takes minutes. The cost is **collecting samples** of each generator.

That side output should be read as **"closest known style, with a confidence"**, not as proof of origin, for four reasons:
- **Open-set gap.** A new generator is silently labelled as a known one (SD1.x → SD2/MJ/SD3, GLIDE → DALL-E 2), and the "unknown" flag is unreliable (0.768). Never show a generator name without a "may be an unlisted generator" caveat.
- **Update lag.** The cue is version-specific style (`020/results/critique.md` §3). A new version, fine-tune, LoRA or style preset may be misattributed until it is collected and added. This was not measured (no version-shift test). Near-twins cannot be separated at all, so names should be reported at family level (e.g. "SD1.x").
- **Attribution ≠ detection.** Real-vs-fake on unseen generators is still capped by the detection-cycle problems: leave-one-generator-out ceiling 0.855, unseen real pipelines 16–19% FPR (`docs/03-final-report.md` addenda). Attribution is only meaningful *after* an image is judged fake.
- **Platform is not a usable feature.** Platform history is lost in laundered pixels, and the detectors do not use it (§4, H-A3).

**Provenance first, forensics second.** Pristine files carry strong evidence that laundering destroys:
- metadata: 0.688 generator / 0.985 platform;
- VAE decoder match: 0.967;
- PRNU against a claimed device's reference photos.

So a sensible system is tiered:
1. Check file-level evidence while it exists: metadata, decoder match, cryptographic provenance such as C2PA content credentials, and watermarks. Content credentials and watermarks were not studied in this cycle and are not in our KB [unverified].
2. Fall back to the laundering-robust style probe.
3. Say "unknown" when neither is confident.

**PRNU answers a different question:** "did *this* device take *this* photo?". It needs reference photos from the claimed device. It suits claim verification, not mass screening. On modern phones it is risky: Samsung default-mode fingerprints show FPR 0.66–1.0 at PCE 60 (`kb/papers/samsung-prnu-arxiv2025.md`), and Apple portrait mode adds shared patterns (`kb/papers/sdnp-tifs2026.md`).

---

## 8. Limitations

- **Sample sizes.**
  - About 24 test images per generator per fold (`020/notes.md`).
  - H-A6 has 108 test images, with 4–10 per "other" generator (`024/results/critique.md` §5).
  - PRNU: 14 devices, 4 same-model sets (9 devices). The FPR is bounded at ≤ 0.0012, not 0.
  - HDR+: 43 burst groups.
- **2010s phones only.** VISION devices are 2010s phones, and the PRNU arm used iPhone 4s–6. Neural-ISP phones, and current WhatsApp/Facebook pipelines, are untested (`021/results/critique.md` §4).
- **Content and prompts.**
  - All fakes come from RAISE captions.
  - Synthbuster prompts were "edited to be photorealistic" (`kb/datasets/synthbuster.md`). Whether the edits differ per generator is [unverified] (prompts.csv was not downloaded).
  - DALL-E 3 server-side prompt rewriting is [unverified] for Synthbuster. For DALL-E 3 (and possibly MJ), "same caption" may not mean the same content.
- **Method scope.**
  - One frozen backbone (DINOv2-base) with linear probes.
  - Detectors were read only at their last layer, and the RINE CLS stack was not probed.
  - Launderings are simulated; no real platform uploads of fakes were made.
- **Unverified mappings.**
  - The RAID gen_k → generator mapping is [unverified].
  - Whether sd-vae-ft-mse equals the VAE that SD1.x/SD2 ship is [unverified].
  - HDR+ codename merging is [unverified].
- **Licences.**
  - Synthbuster: CC BY-NC-SA 4.0 (non-commercial).
  - B-Free SD3/FLUX images: GRIP non-commercial.
  - RAID upstream image licence: [unverified] (research use only).
  - VISION: CC BY-SA 4.0 (`kb/datasets/*.md`, docs/08 §0).
  - Any model trained here inherits these restrictions.

---

## 9. Recommended next steps (ranked)

**Each step requires an owner limit raise: the D-018 stop rule is reached (D-019).** About 14.2 of 20 GB of downloads are used (LOG 2026-10-27 14:26), so about 5.8 GB remain (derived).

1. **Owner's modern-phone photos as the real-side acceptance test.**
   - What: PRNU (same-device, plus WhatsApp copies) and the frozen detector's FPR, on 150–250 photos from a 2021+ phone (README open item 1).
   - Why: it is the single biggest unknown, both for PRNU on neural-ISP phones and for detection FPR.
   - Cost: no download, clean licence, about 1–2 h CPU [estimate].
2. **Version-shift / update-lag test for H-A1.**
   - What: add a native-scale baseline to gate 2, and probe a "same generator, new version" class (a newer MJ/DALL-E version, or SD fine-tunes/LoRAs), as the H-A1 critic requested (`020/results/critique.md` §6).
   - Cost: a new data source within about 5.8 GB, needing owner licence approval; probes take minutes on CPU.
3. **Is the open-set and near-twin failure due to small n?**
   - What: Tier-2 data (Synthbuster +180 per generator ≈ 2.0 GB; B-Free SD3/FLUX +150 each ≈ 0.5 GB; docs/08 §4), with the open-set rule frozen in advance.
   - Cost: about 2.5 GB (NC licence) and CPU hours.
4. **Probe detector *intermediate* blocks, including the pre-registered RINE CLS stack.**
   - What: test whether detector backbones keep generator identity before their last layer (`023/results/critique.md` §2). Descriptive only.
   - Cost: about 3k CLIP-L forward passes on GPU [critic estimate], no download.
5. **A tiered attribution prototype** (file evidence → VAE/PRNU → style probe → "unknown").
   - What: a pre-registered test on a mixed stream of pristine and laundered images, reporting how often each tier answers and how often it is wrong.
   - Cost: CPU hours on existing data.
