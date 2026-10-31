# Final-run report (D-020): what survived the challenge to the "mid-layer rendering-style" claim

**Author:** fable/high, 2026-10-31. **Plan:** `docs/11-final-run-plan.md` (pre-registered 2026-10-29 20:59, before any result). **Inputs:** `experiments/026-final-run/{r1r4,r2r3}/` (the runs), `r5_novelty.md` (literature), `critique.md` + `critic/` (opus/high critic).
**Reading rule.** Where the critic and the run disagree, the critic's reading wins, and the difference is stated. Run numbers are pre-registered or frozen-config outputs. **Every critic number is exploratory and post hoc** (the critic chose block 6 and C = 0.01 after seeing results; its reference is 0.867 against the nested 0.857, about +0.01 optimistic; `critique.md` header).
**Short names.** CRIT = `experiments/026-final-run/critique.md`. R1R4 = `r1r4/results/tables.md` and `r1r4/notes.md`. R2R3 = `r2r3/results/tables.md` and `r2r3/notes.md`. R5 = `r5_novelty.md`. Critic scripts = `critic/c1_cached.txt` ... `c8_kernel.txt`.

## 0. Plain-language vocabulary (used below)
- **Attribution.** Given one fake image, name the generator that made it. Here: 8 generators, about 100 training images each. This is not detection (real vs fake).
- **Laundering.** JPEG compression plus resizing, as a social platform would do. Primary test condition: `jpeg75_resize05` (JPEG quality 75 after a 0.5x resize).
- **Frozen backbone / probe.** A big pretrained vision transformer (ViT) whose weights we do not change, plus a small linear classifier (the probe) trained on its internal features.
- **Block / layer.** A ViT is a stack of 12 (or 24) blocks. "Mid" means blocks at 25-75% depth. "Last" means the final block.
- **bacc.** Balanced accuracy over the 8 generators. Chance is 0.125.
- **Retained.** (bacc after manipulation minus chance) divided by (bacc unmanipulated minus chance). It is measured after retraining the probe on the manipulated images, so it says "the information is still there", not "the original probe uses it".

## 1. Conclusion (one paragraph)
The claim under test was that, after laundering, generator identity lives in mid-depth ViT features as "local rendering style", and that laundering-aware training makes this transfer to launderings never seen in training. **The measurements hold; two of the three interpretations do not.** Mid layers clearly beat the last layer for all five backbones (+0.148 to +0.295 bacc, R1R4), but that effect is already published (Li et al. 2026, R5), so only its size under our protocol is ours. The signal does not need image layout (tile shuffle keeps 0.944 of it after retraining) and is not a global tone curve (equalising the histogram keeps 0.904, critic). It is better described as several redundant multi-scale statistics, including a strong sharpness axis that the laundering-trained probe depends on. "Transfers to unseen launderings" is not supported: the two "unseen" launderings were close to the training family, and genuinely new mild blur or unsharp masking drops the probe to about 0.52 (critic). The extrapolation to 0.92-0.94 at 10,000 images per generator is not licensed. Above all, each class here is one configuration from one operator, so we cannot say whether we identify generators or generator settings, and no real photos were used, so nothing here bears on the project's FPR targets.

## 2. Scorecard of the five pre-registered challenges
Pre-registered verdict (docs/11 rules, from the run) first, then the critic's reading.

| # | Challenge | Pre-registered verdict (source) | Critic's reading (exploratory) |
|---|---|---|---|
| R1 | Generality across backbones | **GENERAL, 4 of 4.** Mid minus last: CLIP-OpenAI +0.249 [0.219, 0.282], CLIP-LAION +0.196, B-Free +0.167, CommFor +0.148; DINOv2-base reference +0.295 (R1R4). Frozen CLIP pair: general (2 of 2). | **Fair, with limits.** Not probe-capacity artefact: block 12 stays 0.48-0.56 across C, post-LayerNorm and RBF/MLP heads (`c3_stats.txt`). RAID shows the same pattern (b5 0.718 vs b12 0.537, n = 400, descriptive). The 4 backbones are not independent (same images, all ViTs); CommFor sees an upsampled 384 input and B-Free a tiled one, so rankings are not like-for-like (CRIT §4). |
| R2 | Mechanism | (a) tile shuffle keeps >= 70%: **confirmed**, 0.944 [0.918, 0.970]. (b) blur s=1.5 keeps < 50%: **refuted**, 0.972 [0.952, 0.991]. Mid > last persists under shuffle: **confirmed**, +0.241 [0.211, 0.271] (R2R3). | **Numbers right, wording too strong.** "Retained" is sufficiency after retraining. The frozen probe falls to 0.499 under 32-px shuffle (`c1_cached.txt` B). The blur result depends on dose: s = 3 / 6 / 12 / 24 retain 0.851 / 0.716 / 0.652 / 0.561 (`c5_manip.txt`). Read it as "multi-scale", not "low-pass is enough". |
| R3 | Laundering stress map | Smooth decay: **refuted narrowly** (max step drop 0.204 [0.189, 0.234] vs 0.20 rule, at resize 0.35 to 0.25). Extended augmentation moves the break by >= 1 step: **confirmed on thin evidence** (1 of 30 cells below 0.5) (R2R3). | **Do not headline either way.** The grid is JPEG x resize and includes the training cell, so it maps the training family, not general robustness. The resize axis mixes scale, field of view and upsampling; 0.204 vs 0.20 is a coin flip on a confounded step (CRIT §4). |
| R4 | Scaling with data | Bacc 0.691 / 0.778 / 0.832 / 0.853 / 0.861 for n = 15 / 30 / 60 / 90 / 102 (DINOv2 block 6). No plateau flagged (b = 0.661 [0.394, 1.081]). Extrapolated 0.911 (n=1,000), 0.923 (n=10,000) (R1R4 notes). | **Extrapolation not licensed.** Three curve forms fit within noise and give 0.860 / 0.923 / 0.980 at n = 10,000. "No plateau" is a weak test (any rising curve has b > 0). Only licensed: more data helps up to n = 102 (+0.17 from 15, +0.03 from 60), "beyond that, measure" (`c3_stats.txt`, CRIT §4). |
| R5 | Novelty (literature) | (a) mid-layer attribution advantage: **KNOWN qualitatively** (Li arXiv 2607.12052; Cioni ECCV-W 2024). (b) transfer to held-out launderings: **NOVEL (not found)**. Also checked (not in docs/11): (c) generic beats detector embeddings: "novel, not found"; (d) patch-shuffle analysis: "partially known" (R5). | (b) becomes "not found, and our own version is weak". (c) is **retracted at mid blocks** (below). Both R5 results are absence in 11 searches; pages were read through a summarising tool, so quotes are [unverified] against the PDFs (R5 header). |

## 3. What the signal is, and is not
### 3.1 The mid-layer advantage is general, and already known
- Five ViTs, one pattern: best mid block 0.839-0.896 against last block 0.562-0.735 (R1R4). The block chosen by inner cross-validation never left the mid range (R1R4 notes).
- Nested selection bias is small: nested vs oracle block differs by at most 0.011 (`c3_stats.txt`, CRIT §4).
- Li et al. (arXiv 2607.12052, preprint) report that attribution accuracy "peaks at intermediate representation levels" for CLIP and DINOv2, also under JPEG/blur/resize at test time (R5 (a)); RINE shows the same for detection (`kb/papers/rine-eccv2024.md`). Our contribution is only the effect size with a trained probe, caption-family folds and laundering-trained probes.

### 3.2 Layout is not needed; tone is not the explanation
- **Tile shuffle** (32-px tiles, probe retrained): bacc 0.816, retained 0.944. Mid-over-last survives (+0.241) (R2R3). The same style of test on RAID: 32-px shuffle 0.675 vs 0.693 unshuffled at b6 (critic, descriptive, `c6_raid.txt`).
- **Histogram equalisation per channel** (retrained): 0.796, retained 0.904 [0.873, 0.935] (critic, `c5_manip.txt`). Local mean/std per 32-px tile keeps 0.862; per-image mean/std keeps 0.945.
- **Gamma 0.7** leaves the frozen probe at 0.871 after laundering (clean 0.922) (`c5_manip.txt`). A global tone curve is **rejected**.
- Hand-built features (critic, `c4_hist.txt`): all tone statistics 0.466-0.516, all texture statistics 0.512-0.521, tone plus texture 0.573-0.633, against 0.867 for DINOv2 block 6. The earlier M1 baseline (0.425) understated the tone component; all-tone HGB (0.516) is close to DINOv2's last block (0.56).

### 3.3 Redundant cues and the sharpness axis (critic, exploratory)
- **At least two partly independent cues.** Coarse tone and colour alone give about 0.52-0.54. With tone, colour, layout and fine detail all removed (grey + equalised + 14-px shuffle + blur s=1.5), 0.563 [0.535, 0.591] remains. Together they give 0.867 (`c5_manip.txt`).
- **Sharpness axis.** The frozen laundering-trained probe collapses under changes of sharpness: Gaussian blur s=2 gives 0.525 [0.502, 0.550]; unsharp mask (r2, 150%) gives 0.524 [0.500, 0.550]. Pre-blur s = 0.5 / 1 / 2 / 3 gives 0.855 / 0.789 / 0.525 / 0.400, a monotone dose-response (`c5_manip.txt`, `c7_sharp.txt`).
- **The information is there.** Adding blur s=2 and unsharp 150% to training recovers pre-blur s=1 to 0.850 and unsharp 100% to 0.861, with the primary condition still 0.865 (`c7_sharp.txt`). Transfer between directions stays partial: blur training lifts unsharp from 0.524 to 0.672.
- **Fragility.** Nominal s=1.5 blur after laundering gives 0.816 with scipy and 0.674 with PIL, though the two filters differ by only 0.38/255 per pixel; the cause was not traced (`c8_kernel.txt`). Whether real platforms sharpen after resizing is [unverified].
- Net description: "redundant multi-scale image statistics: coarse tone and colour, a sharpness axis, and layout-independent local structure" (CRIT §2). "Local texture" alone (the docs/11 reading) is too simple, and the run's own caveat agrees: high-pass loses the most (retained 0.854), so some information sits in low/mid frequencies or colour (R2R3 notes).

### 3.4 Retrained-sufficient vs frozen-probe-uses
Two different questions, easily confused:
- **Sufficient (retrained):** after training on manipulated images, is the information still there? Shuffle 0.816, equalised 0.796, blur 0.837, grey 0.797, high-pass 0.750 (R2R3).
- **Used (frozen):** does the probe we actually deploy rely on it? Applying the unmanipulated probe: shuffle 0.499, equalisation 0.527, grey + equalised 0.476, 14-px shuffle 0.357 (`c1_cached.txt` B, `c5_manip.txt`).
- Part of the frozen drop may be distribution shift, so the information is not proven gone. But "shuffle retains 0.944" must never be quoted as "the probe ignores layout". All sufficiency statements in this report are about the retrained probe.

### 3.5 Not a two-batch artefact (critic)
- Synthbuster-only 6-way: 0.867 (chance 0.167). SD3 vs FLUX, both from the same batch: 0.960. A Synthbuster-vs-other detector scores 0.935 but does not generalise with a generator held out. Cross-batch share of errors is 28% against about 36% if uniform (critic's calculation). 269 of 542 caption families mix batches (`c1_cached.txt` C-D, `c6_raid.txt`, CRIT §3).
- Weakest class: Midjourney (recall 0.683). SD2 and SDXL confuse each other (6 each way). The mid-over-last gain is broad: all 8 classes, +0.21 (SD2) to +0.39 (SDXL).

## 4. Retractions (what this run overturns)
1. **`docs/10-attribution-report.md` §6, "plausibly new" item 1** (mid-layer advantage for attribution under laundering): downgraded to **known**. Li et al. 2607.12052 and Cioni 2408.09153 precede it, and RINE covers detection (R5). What remains ours: the effect size (b8 minus b12 = +0.254 in docs/10; +0.295 nested here) under a trained, laundering-trained probe. §6's sentence "not reported in the ~30 attribution papers we read" was true of our reading list only.
2. **docs/10 §6 item 2** ("a laundering-trained probe transfers to unseen launderings", 0.866 / 0.813), and the matching statement in the LOG 18:41 entry ("holds on unseen launderings"): **not supported.** `screenshot_sim` has no JPEG and a clean-only probe already scores 0.875 on it (T-mix 0.899, gain +0.024); `mavali_medium` (resize 0.75 + q50) is only 20 quality units outside the training family, and clean + JPEG-only training gives 0.818 (vs T-mix 0.828). Without JPEG in training there is no transfer to JPEG (0.51-0.53) (`c1_cached.txt` A, CRIT §1). Always quote the clean-only 0.875 next to screenshot_sim. R5 (b) "NOVEL" becomes "not found, and our own version is weak".
3. **docs/10 §6 item 4, "detector embeddings worse than generic DINOv2 (-0.118)", the "generic beats detectors" statement (R5 (c)), and the H-A5 interpretation.** The 023 gap came from reading detectors at their last layer only, as the 023 critic warned. At mid blocks: CommFor beats DINOv2-base by +0.026 [0.006, 0.047]; CLIP-OpenAI minus CommFor +0.014 [-0.007, 0.034]; CLIP-LAION minus CommFor +0.003 [-0.017, 0.023]; CLIP-OpenAI minus B-Free +0.057 [0.035, 0.078] (tiling caveat) (R1R4, CRIT §4). **The pre-registered H-A5 label stays KILL** (it was decided by its own protocol and still counts as failure 1 of 3 under D-019); only the explanation changes.
4. **"Rendering style = local texture".** Replaced by the multi-cue description in §3.3. The 020 critic's "the probe reads mid-level rendering style, not semantics" is kept only as "the last block is poor and the mid block is not semantic"; it is not a proof of what is read.
5. **R4 extrapolation (0.92-0.94 at n = 10,000) and "no plateau".** Replaced by "0.86-0.98 depending on curve form (DINOv2); measure it" (§2). The reported CI [0.872, 0.984] covers sampling noise only; the reported fit itself asserts a plateau (0.926, 0.939); top-level subsamples overlap heavily (90 of 102) (`c3_stats.txt`).
6. **R3 "1 of 30 cells below 0.5" as robustness, and the "cliff".** The grid maps the training family (5 cells are below 0.6).
7. **R2 "retained" as "the probe ignores X".** See §3.4.
8. **Pending edits requested by the critic and not made by this report** (I edit no other file): README and LOG wording of the transfer claim; correct `r5_novelty.md` (b) and (c); r2r3 notes ("retained" = after retraining; quote the frozen column); r1r4 notes (replace the extrapolation headline with 0.86-0.98); log the selftest-commit deviation (§7).

## 5. Claims ranked
**Scope statement.** All 1,020 images in this run are fakes (8 generators). **No real images were used, so nothing here bears on the FPR <= 1% (fresh phone photos), AUROC >= 0.95 (laundered) or RAID-transfer acceptance targets.** Open-set attribution also remains unusable (020: msp mean AUROC 0.768, `LOG.md` 18:24 entry).

### (a) Solid and new-ish (literature check still pending; verify before any claim)
- Frozen ViT features stay sufficient for laundered attribution after tile shuffle (0.816; RAID 0.675) with mid-over-last persisting (+0.241), and after equalisation (0.796). Sufficiency after retraining only (R2R3; critic).
- Exploratory, needs pre-registration: the cue decomposition (tone and colour about 0.52, stripped remainder about 0.56, combined 0.867) (`c4_hist.txt`, `c5_manip.txt`).
- Exploratory, **literature not searched by anyone**: laundering-trained probes have a sharpness axis (mild blur or unsharp gives 0.52; training on them recovers 0.82-0.86 at no primary cost) (`c7_sharp.txt`).
- Why "pending": R5 is 11 searches, not exhaustive, with pages read via a summariser (R5 header); the critic did no new literature search (CRIT §5).

### (b) Solid but known
- Mid beats last, 5 of 5 backbones including the reference, +0.148 to +0.295; RAID +0.181 (CRIT §3); robust to probe capacity. Li 2607.12052 precedes us.
- Laundering-aware training beats clean training on JPEG conditions (0.867 vs 0.406; `c1_cached.txt` A). Augmentation restoring robustness is standard (`kb/papers/pose-cvpr2023.md`, per R5).
- Closed-set bacc 0.839-0.896 for 8 known generators with about 100 images each (R1R4). A solid baseline, not state of the art (docs/10 §6).
- Backbone differences are small: CLIP-OpenAI over DINOv2 +0.040 [0.020, 0.062], CLIP-LAION +0.029, CommFor +0.026 (R1R4).

### (c) Not supported; do not claim
- "Transfers to unseen launderings" (§4.2).
- "Generic features beat detectors" at matched depth (§4.3).
- R3 as general robustness, and the "cliff".
- "0.92-0.94 at n = 10,000" and "no plateau".
- "A trivial tone curve explains it" (rejected in the other direction too: it is not tone alone).
- "The probe ignores layout" (§3.4).

## 6. Handoff for future research with more compute and resources
**The single most important untested hole: each class is one configuration from one operator** (sampler, steps, guidance, upscaler, prompt edits; per-generator details are [unverified]). With this data, "generator" and "generator + configuration" cannot be separated (CRIT §3). Every other result in this report may be reading the settings, not the model. Items below follow the critic's order (CRIT §5, exploratory); the data and compute figures are my estimates unless a file is cited.

1. **Configuration vs architecture (highest value).**
   - Generate locally from one shared prompt list: 3-4 open models x about 5 configurations (sampler, steps, guidance, VAE/upscaler, LoRA) x about 500 images each. That is 7,500-10,000 images, about 7-10x the current 1,020 (arithmetic).
   - **Falsifier:** leave-configuration-out bacc drops by more than 0.20, so the probe attributes settings, not models.
   - Compute [estimate, not measured]: feature extraction for five backbones took about 35 minutes for the current 1,020 images (R1R4 notes), so linear scaling suggests roughly 4-6 hours; image generation time depends on the models and the GPU and is [unverified].
2. **Generators as the unit.**
   - At least 30 generators, with CIs over generators (today the 8 generators are fixed, and the CIs resample caption families only; CRIT §4).
   - Data [estimate]: about 100 images per generator (R4 shows +0.03 from 60 to 102 per generator), so about 3,000 images.
   - **Falsifier:** the mid-over-last CI over generators includes 0.
3. **Real laundering chains with whole-family hold-out.**
   - Platform round-trips, real screenshots, phone recapture, super-resolution, denoisers, sharpening, other codecs (WebP q50 alone costs the frozen probe a lot natively: 0.682, `c5_manip.txt`).
   - Pre-register blur + unsharp augmentation and its clean cost.
   - Hold out an entire family at a time (leave-family-out); today "unseen" means near the training family.
   - **Falsifier:** leave-family-out bacc under 0.6 with broad augmentation.
   - Data [estimate]: the existing fakes pushed through the real chains; no new generation needed, but account access and manual capture time are [unverified].
4. **Measure R4, do not fit it.**
   - Synthbuster has 9 generators at 1,000 images each (`kb/datasets/synthbuster.md`), about 9,000 images; the file size and the D-015 ceiling check are not done here, and the download needs owner approval.
   - This separates the 0.86 / 0.91 / 0.95 curve forms cheaply at n up to 1,000.
5. **Pre-register the frozen-vs-retrained split** and the cue decomposition, including whether a probe trained on stripped input transfers to natural images.
6. **Keep real images in scope.** Test whether a mid-block, sharpness-augmented feature improves leave-one-generator-out detection. Otherwise this stays an attribution side-result with no bearing on the acceptance targets.

**Likely a waste (critic):** more backbones on the same 8 generators; finer JPEG x resize grids; more n on the same configurations before item 1; CLIP-vs-DINOv2 margins of 0.01-0.04; more tile sizes; novelty claims from search snippets; more curve forms fitted to 5 points.

**Open owner items (unchanged by this run, `README.md` "Open items"):**
- 150-250 photos from a 2021+ phone, plus WhatsApp copies, in `data/raw/owner_phone/` (git-ignored). This is the acceptance test for any detector (H3) and the only route to evidence on the FPR target. Modern neural-ISP photos are not covered by any result here.
- D-013 (`DECISIONS.md`, PROPOSED): whether the `fov512` protocol is primary. The R1-R4 features used the 004 fov512 crop (R1R4 notes), so this run already assumes it. The README says the report recommends fov512.

## 7. Process notes and deviations
None changes a reported number (CRIT §4 for items 1-3, 6).
1. **Selftests were committed with the results, not before.** docs/11 required selftests committed before real-data results. They first appear in the result commits (e9b6933, 04577ed), so the code freeze rests on file timestamps only: r2r3 config 02:10, first real data 02:18; r1r4 config 02:16, extraction 02:21. To be logged by the lead.
2. **Cliff, shift and plateau rules live in `config.json`, not in docs/11.** Cliff = max single-step drop > 0.20; break = bacc < 0.5; shift = median over slices; plateau = lower CI of b <= 0.05 (R2R3 notes, R1R4 notes). They were fixed before real data by timestamp.
3. **The plateau rule changed during the selftest:** the first draft was <= 0.001, the flat-curve selftest case showed that was too strict, and it became <= 0.05 before any real result (R1R4 notes, deviation 4). The critic judges the difference irrelevant: b lower bounds were 0.394 and 0.483.
4. **D-008 (GPU vs CPU) was not re-run for the critic's new operations** (blur, unsharp, equalisation, WebP and others). D-008 passed for the run's own pipeline: max scaled feature diff 4.0e-5 to 9.4e-5 per backbone, and max |dlogit| 3.0e-5 for the R2/R3 variants (R1R4, R2R3 gates). The critic's cached features match the r2r3 cache to 1.9e-6 (`critic/c2_extract.log`).
5. **Backbone change before any result (docs/11 addendum).** The DINOv2-small/large download was not approved, so R1 used DINOv2-base (reference) plus OpenAI CLIP-L, LAION CLIP-L, B-Free and Community Forensics ViT-S. R1 verdict rule unchanged. A second consequence: only one non-CLIP generic family was available, and the two detector backbones are fine-tuned, possibly on related generators (023 overlap flags) (R1R4 caveats).
6. **Input-handling deviations.** B-Free's wrapper tiles a 224 crop about 2.25x; Community Forensics resizes to 440 and crops to 384, so its 256-px laundered images are upsampled; R3 cells at resize 0.35 and 0.25 are upsampled to 224 (R1R4 notes, R2R3 notes). Rankings across backbones are therefore not like-for-like, and the R3 cliff is confounded.
7. **Other facts to keep.** The "all" training size is 102 per generator, not about 96 (CRIT §4). The 32-px shuffle does not align with the 14-px patch grid, adding seam artefacts that the 0.944 retained survives (R2R3 caveats). One cosmetic report-line reword after the r2r3 results, with `results.json` verified identical (R2R3 notes).
8. **Critic procedure.** Critic features are in `critic/cache/` (git-ignored; rebuild with `critic/c2_extract.py`). The critic edited no shared file and found no leak and no bug that changes a number (CRIT header).
9. **Grounding.** Every number above comes from the cited file. Anything that depends on the PDFs of Li 2607.12052, Cioni 2408.09153 or other R5 sources is [unverified] until read in full. Real-platform sharpening and the per-generator configuration details are [unverified].
