# Critique of 026-final-run (D-020, R1-R5): opus/high critic, 2026-10-31
**Verdict.** I found no leak and no bug that changes a reported number. The measurements are sound, but three interpretations do not hold:
- "Transfers to unseen launderings" is not supported as worded. It is interpolation inside the JPEG/resize neighbourhood.
- "Rendering style = local texture" is too simple. It is not a global tone curve, but it is several redundant cues, and the trained probe leans hard on a sharpness axis.
- The R4 extrapolation (0.92-0.94) is not licensed by the data.

**My checks** are in `critic/` (c0-c8). Unless stated: DINOv2-base block 6, CLS+mean, fixed C = 0.01, same rows and folds, T-mix training, test at jpeg75_resize05 (j75r05). Block and C were picked after seeing the run's results, so this reference is 0.867 against the nested 0.857 (about +0.01 optimistic). **All critic numbers are exploratory, not pre-registered.** Parity with the r2r3 cache: scaled max diff 1.9e-6 (`c2_extract.log`). D-008 was not re-run for my new operations.

## 1. Is the laundering transfer real? (`c1_cached.txt`, `c5_manip.txt`, `c7_sharp.txt`)
**The two "unseen" launderings are not another family (`lib/perturb.py`).** T-mix covers resize 0.5-1.0 and q70-100. `mavali_medium` is resize 0.75 + q50: only the quality is outside, by 20 units. `screenshot_sim` is a bilinear resample {0.5, 0.67, 1.25} + a small crop, with **no JPEG**. A clean-only probe already scores 0.875 on it (T-mix 0.899, gain +0.024), so it is not evidence of laundering transfer.

| training set (bacc) | j75r05 | screenshot | mavali | q95,f0.5 | q45,f1 | q30,f1 |
|---|---|---|---|---|---|---|
| T-mix | 0.867 | 0.899 | 0.828 | 0.873 | 0.795 | 0.705 |
| clean only | 0.406 | 0.875 | 0.382 | 0.589 | 0.420 | 0.402 |
| clean + screenshot (resample, no JPEG) | 0.530 | 0.913 | 0.514 | 0.692 | 0.532 | 0.509 |
| clean + JPEG only (q75, q45, f=1) | 0.790 | 0.876 | 0.818 | 0.829 | 0.837 | 0.807 |
| clean + mild JPEG + resize (q95; f0.75, 0.5) | 0.819 | 0.895 | 0.760 | 0.906 | 0.744 | 0.647 |

No JPEG in training means no transfer to JPEG (0.51-0.53). With any JPEG, transfer along the q and f axes is decent (JPEG-only reaches j75r05 at 0.790 [0.766, 0.814]). Mavali needs little of T-mix: clean + JPEG-only gives 0.818 [0.793, 0.841].

**Genuinely different operations (frozen T-mix probe, never trained on them):**
| operation | native (clean 0.935) | then j75r05 (0.867) |
|---|---|---|
| gamma 0.7 / 1.4 | 0.922 / 0.922 | 0.871 / 0.839 |
| warm cast; contrast + saturation x1.3 | 0.919; 0.903 | 0.820; 0.801 |
| rotation 5 degrees | 0.923 | 0.851 |
| noise s=8 + q90; WebP q50 | 0.770; 0.682 | 0.841; 0.836 |
| median 3; nearest x0.5 and back | 0.656; 0.680 | 0.849; 0.844 |
| **Gaussian blur s=2 on the 512 image** | 0.692 | **0.525 [0.502, 0.550]** |
| **unsharp mask (r2, 150%)** | 0.693 | **0.524 [0.500, 0.550]** |

The last column is partly easy: that step returns the image to a trained condition. Dose-responses are monotone: pre-blur s = 0.5/1/2/3 gives 0.855/0.789/0.525/0.400; unsharp 50/100/150% gives 0.780/0.632/0.524. Errors pile into one class (blur: 50-65% to dalle2; unsharp: 52% to sd3_large). **The trained probe has a strong global sharpness axis.** It is also fragile: nominal s = 1.5 after laundering gives 0.816 with scipy and 0.674 with PIL, though the filters differ by only 0.38/255 per pixel (`c8_kernel.txt`; cause not traced).

**The fix is cheap.** T-mix + blur s=2 + unsharp 150%: pre-blur s=1 0.850, unsharp 100% 0.861, primary still 0.865. The information is there; the frozen probe was not forced to ignore sharpness. Across directions transfer stays partial: blur training gives unsharp 0.672 (from 0.524); unsharp training gives blur 0.706 (from 0.525).

**Verdict: does not survive as worded.** Use: "robust within the JPEG/resample neighbourhood and to global photometric shifts and small rotation; not robust to sharpness-changing operations unless they are in training." Real platforms may sharpen after resizing [unverified]. R5(b) "NOVEL" becomes "not found, and our own version is weak".

## 2. Is it trivial tone, colour or histogram? (`c4_hist.txt`, `c5_manip.txt`) The key question
**No, global tone does not explain it. It is a real component, and M1 0.425 understated it.** Statistics of the same 224 input (LR C chosen on test, which favours the trivial explanation):
| features | LR | HGB |
|---|---|---|
| mean + std RGB (6-d) / luminance histogram | 0.304 / 0.313 | 0.318 / 0.346 |
| RGB histograms | 0.386 | 0.443 |
| all tone (histograms, 3-D colour, moments; 710-d) | 0.466 [0.439, 0.493] | **0.516** |
| all texture (gradient, Laplacian, LBP, radial PSD) | 0.521 | 0.512 |
| tone + texture (1,078-d) | 0.573 | **0.633** |
| DINOv2 block 6 | **0.867** | |

All-tone HGB (0.516) is close to DINOv2's **last block** (0.56); read "the last block is poor" against that.

**Manipulating the input (T-mix retrained per manipulation; last column = unmanipulated probe, frozen):**
| manipulation | bacc | retained | frozen |
|---|---|---|---|
| mean/std per image | 0.826 | 0.945 | 0.702 |
| histogram equalisation per channel | 0.796 | 0.904 [0.873, 0.935] | 0.527 |
| grey + equalised | 0.749 | 0.841 [0.808, 0.876] | 0.476 |
| local mean/std per 32-px tile | 0.765 | 0.862 | 0.525 |
| 14-px tiles aligned to the patch grid | 0.835 | 0.958 | 0.357 |
| grey + equalised + 14-px shuffle | 0.681 | 0.749 | 0.274 |
| + blur s=1.5 (all removed) | 0.563 [0.535, 0.591] | 0.590 | 0.224 |
| phase scrambled / amplitude whitened | 0.588 / 0.617 | 0.624 / 0.663 | 0.251 / 0.197 |
| blur s = 3 / 6 / 12 / 24 | 0.756 / 0.656 / 0.609 / 0.541 | 0.851 / 0.716 / 0.652 / 0.561 | 0.470 / 0.184 / 0.142 / 0.134 |

1. **At least two partly independent cues.** Coarse tone and colour alone give about 0.52-0.54 (histograms; s=24 blur). With tone, colour, layout and fine detail removed, about 0.56 remains. Together: 0.867.
2. **R2's "retained" measures sufficiency after retraining, not what the probe uses.** The frozen probe falls to 0.50 under 32-px shuffle (`c1_cached.txt` B), 0.357 under 14-px shuffle, 0.527 under equalisation. Part of that may be distribution shift, so the information is not proven gone. But "shuffle retains 0.944" must not be read as "the probe ignores layout".
3. **"Blur retains 0.972" depends on the dose.** s=1.5 after a 2x downscale is mild. Decay is gradual with no single carrying band. Prediction (b) is correctly refuted, but what follows is "multi-scale", not "low-pass is enough".

**Replacement wording:** "redundant multi-scale image statistics: coarse tone and colour, a sharpness axis, and layout-independent local structure." "Just a global tone curve" is **rejected** (gamma 0.7 leaves 0.871; equalisation retains 0.904).

## 3. Generator or dataset batch? (`c1_cached.txt` C-D, `c6_raid.txt`, `c0_census.py`)
**A two-batch explanation is not supported.** Synthbuster-only 6-way: 0.867 (chance 0.167). sd3 vs flux, both GRIP: 0.960. A Synthbuster-vs-GRIP detector scores 0.935, but with a generator held out it does not generalise: held-out sd3 is called GRIP 15.3%, flux 19.3%, Synthbuster generators 1.7-47.5% (dalle3 highest). Cross-batch share of errors is 28%, against about 36% if uniform (my calculation). Captions are shared: 269 of 542 families mix batches.
**Confusions:** midjourney is weakest (recall 0.683; 12 errors to flux, 10 to firefly); SD2 and SDXL confuse each other (6 each way). The b6-over-b12 gain is broad: all 8 classes, +0.21 (SD2) to +0.39 (SDXL).
**RAID replicates the pattern** (n = 400, 4 anonymous generators, chance 0.25, jpeg(75); oracle block, no CI, descriptive): b1 0.570, b4 0.705, **b5 0.718**, b8 0.647, b12 0.537, so +0.181. At b6 (0.693): 32-px shuffle 0.675, grey + equalised 0.628, blur s=3 0.637, s=6 0.522, all removed 0.542. 3-way without gen_0: 0.670 (chance 0.333).
**The real hole, not testable here.** Every class is **one configuration from one operator** (sampler, steps, guidance, upscaler, prompt edits; per-generator edits still [unverified]). "Generator" and "generator + configuration" cannot be separated with this data.

## 4. Selection, CIs, R4 and deviations (`c3_stats.txt`)
- **Block selection: no material bias.** Nested vs oracle differs by at most 0.011 (CLIP-LAION 0.886 vs 0.897); nested-all equals nested-mid.
- **Mid-over-last is not a probe artefact.** Block 12 stays low: LR C 0.001-10 gives 0.537-0.562; post-LayerNorm CLS 0.481-0.491; PCA-200 + RBF / MLP 0.477 / 0.499 (b6: 0.838 / 0.828).
- **CIs are adequate for what they cover.** Six new fold seeds: b6 mean 0.860, SD 0.0045; gap 0.288-0.320; CI half-width about 0.02. They do **not** cover generator sampling (8 fixed), configuration or dataset. "4 of 4 backbones" are not independent: same images, all ViTs.
- **R1 "general" is fair**, but CommFor sees an upsampled 384 input and B-Free a tiled one, so backbone rankings are not like-for-like.
- **"Generic beats detectors" (R5(c)) must be retracted at mid blocks.** CLIP-OpenAI - CommFor +0.014 [-0.007, 0.034]; CLIP-LAION - CommFor +0.003 [-0.017, 0.023]; DINOv2 - CommFor **-0.026** [-0.047, -0.006]; CLIP-OpenAI - B-Free +0.057 [0.035, 0.078] (tiling caveat). The 023 KILL stands by its letter, as its critic warned: it probed the last layer only.

**R4: the extrapolation is not licensed.** Three forms fit within subsample noise:
| form | RMSE (DINOv2) | DINOv2 n=1,000 / 10,000 | CLIP-OpenAI n=10,000 |
|---|---|---|---|
| exponential + c | 0.0036 | 0.860 / 0.860 | 0.897 |
| power + c (reported) | 0.0006 | 0.911 / 0.923 | 0.938 |
| pure power | 0.0037 | 0.948 / 0.980 | 0.990 |

The honest range at n = 10,000 is **0.86-0.98 (DINOv2)**; the reported CI [0.872, 0.984] covers sampling noise only. "No plateau" is a weak test: any rising curve has b > 0, and the reported fit itself asserts a plateau (0.926, 0.939). Top-level subsamples overlap heavily (90 of 102). **It licenses only:** "these 8 generators, same source: more data helps up to n = 102 (+0.17 from 15 to 102, +0.03 from 60 to 102); beyond that, measure." Nothing about new generators or versions.

**R3 maps the training family.** The grid is JPEG x resize and includes the training cell, so "1 of 30 below 0.5" is not general robustness (5 cells are below 0.6). The resize axis mixes scale, field of view and upsampling. "Cliff" at 0.204 [0.189, 0.234] vs 0.20 is a coin flip on a confounded step. Do not headline either way.

**Deviations (none changes a number).** Cliff, shift and plateau rules live in config.json, not docs/11; timestamps support "before real data" (r2r3 config 02:10, first real data 02:18; r1r4 config 02:16, extraction 02:21). Plateau tolerance 0.001 vs 0.05 is irrelevant (b lower bound 0.394, 0.483). **docs/11 asked for selftests committed before real-data results; they first appear in the result commits (e9b6933, 04577ed), so the freeze rests on file timestamps only. Log this.** n "all" is 102, not about 96.

## 5. Bottom line
**(a) Solid and new-ish** (relative to R5's non-exhaustive search; verify before any claim):
- Frozen ViT features stay sufficient for laundered attribution after tile shuffle (0.816; RAID 0.675) with mid over last persisting (+0.241), and after equalisation (0.796). Sufficiency only.
- Exploratory, needs pre-registration: the cue decomposition (tone and colour about 0.52, stripped remainder about 0.56, combined 0.867).
- Exploratory, literature **not** searched by me: laundering-trained probes have a sharpness axis (mild blur or unsharp gives 0.52; training on them recovers 0.82-0.86 at no primary cost).

**(b) Solid but known:** mid-layer > last (4/4, +0.148 to +0.295, RAID +0.18, robust to probe capacity; Li 2607.12052 precedes us, so only the effect size is ours). Laundering-aware training beats clean training on JPEG conditions (0.867 vs 0.406). Closed-set bacc 0.839-0.896 for 8 known generators with about 100 images each. Backbone differences are small (CLIP over DINOv2 +0.03 to +0.04).

**(c) Not supported, do not claim:** "transfers to unseen launderings"; "generic features beat detectors" at matched depth; R3 as general robustness, and the "cliff"; "0.92-0.94 at n = 10,000" and "no plateau"; "trivial tone explains it". Nothing here bears on detection: no real images, so **no evidence on the Phase 5 targets**. Open-set remains unusable (020: 0.768).

**Guidance for more resources (ordered):**
1. **Configuration vs architecture (highest value).** Generate locally from one prompt list: 3-4 open models x about 5 configurations (sampler, steps, guidance, VAE/upscaler, LoRA), about 500 images each. **Falsifier:** leave-configuration-out bacc drops by more than 0.20, so it attributes settings, not models.
2. **Generators as the unit.** At least 30, with CIs over generators. **Falsifier:** the mid-over-last CI over generators includes 0.
3. **Real chains, whole-family hold-out.** Platform round-trips, real screenshots, phone recapture, super-resolution, denoisers, sharpening, other codecs. Pre-register blur + unsharp augmentation and its clean cost. **Falsifier:** leave-family-out bacc under 0.6 with broad augmentation.
4. **Measure R4, do not fit it.** Synthbuster has 1,000 per generator (kb/datasets/synthbuster.md). This separates 0.86 / 0.91 / 0.95 cheaply; the download needs owner approval.
5. **Pre-register the frozen-vs-retrained split** and the cue decomposition, including whether a probe trained on stripped input transfers to natural images.
6. **Keep real images in scope.** Test whether a mid-block, sharpness-augmented feature improves LOGO detection. Otherwise this stays an attribution side-result.

**Likely a waste:** more backbones on the same 8 generators; finer JPEG x resize grids; more n on the same configurations before item 1; CLIP vs DINOv2 margins of 0.01-0.04; more tile sizes; novelty claims from snippets; more curve forms on 5 points.

## 6. Requests to the lead (I edited no shared file)
- README, LOG, docs/10: replace the transfer claim (section 1); quote clean-only 0.875 next to screenshot_sim every time. Correct `r5_novelty.md` (b) and (c).
- r2r3 notes: "retained" means after retraining; quote the frozen column. r1r4 notes: replace the extrapolation headline with 0.86-0.98.
- Log the selftest-commit deviation. `critic/cache/` is git-ignored (features kept, base arrays removed; rebuild with `critic/c2_extract.py`).
