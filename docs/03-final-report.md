# Phase 5 final report: detecting diffusion-generated images in the wild

**Date:** 2026-10-25. **Author:** Phase-5 agent (fable/high), for the lead and the owner.
**Status:** final report of findings. **No method passed its pre-registered tests.** Section 5 gives a *recommended* configuration. It is a recommendation drawn from our evidence, not a validated method.

**Grounding rule.** Every number comes with the path of the result file, critique or KB entry it was taken from. Paths are relative to the repo root. Anything that cannot be traced is marked [unverified]. Arithmetic done for this report, and not present in any result file, is marked "(derived)".

---

## Terms used in this report (each explained once)

| Term | Meaning |
|---|---|
| **Real / fake** | A camera photo vs an image made by a diffusion generator. "Positive" = fake. |
| **AUROC** | The chance that a random fake gets a higher "fake" score than a random real. 0.5 = coin flip, 1.0 = perfect. It measures ranking, not a yes/no decision. |
| **FPR** | False-positive rate: the share of real photos wrongly called fake. The Phase-5 target is ≤ 1%. |
| **TPR@1%FPR** | The share of fakes caught when the threshold is set so that 1% of reals are flagged. |
| **Threshold / calibration** | The cut-off score above which an image is called fake. "Calibrating" = choosing it from real photos only. |
| **Laundering** | Processing that hides traces: resizing, JPEG re-saving, social-media upload, screenshots. Our main test condition is `jpeg75_resize05` (shrink to half size, then JPEG quality 75). `social_sim` is milder (random 0.7–1.0 resize + JPEG 70–100). `screenshot_sim` and `mavali_medium` were never used for training: they are "held-out launderings". Both classes are always processed identically (D-009). |
| **Unseen generator / LOGO** | Leave-one-generator-out: each generator is scored by a model that never saw it. FLUX was never used for training or tuning at all. |
| **Probe** | A simple linear classifier (logistic regression) on top of frozen features from a pretrained network. |
| **DINOv2, block 8 / last block** | DINOv2-base is a pretrained vision network (Apache-2.0) with 12 layers ("blocks"). **B0**, our baseline, is a probe on the *last* block's features. "Block 8" (`F_b8`) is a probe on a *middle* layer's features. |
| **`orig` vs `fov512`** | Two ways to cut a 512-px square from an image. `orig` (D-009): crop the centre with no resizing. `fov512` (D-013): first resize the short side to 512, then crop, so real and fake crops show a similar field of view (zoom). |
| **Real source / pipeline** | Where the reals come from and how they were processed: VISION (old phone photos, plus their WhatsApp and Facebook copies), RAISE (DSLR photos resized by the B-Free authors), OI (Open Images, Flickr photos). |
| **Caption twin** | Each Synthbuster/B-Free fake was generated from the caption of one RAISE photo. That photo is its "twin". Twins share content. |
| **Pre-registration** | Writing the pass/fail rule in `docs/04-phase4-plan.md` (and committing it) *before* seeing the result. |
| **Critic** | An independent opus/high reviewer who re-checks each result and its reading. |
| **Pilot vs official** | Pilots ran on partial data while a data server was down. They never count as tests of a hypothesis. Official runs use the full data (Section 2.4). |
| **C2ST** | "Classifier two-sample test": train a classifier to tell two image sets apart; its AUROC measures how different the sets are. |

---

## 1. Executive summary

### What we set out to do
Find a new method, or a measurable improvement, for detecting diffusion-generated images. The improvement had to hold on **unseen generators** and **in-the-wild images** (JPEG, resizing, screenshots, modern phone photos). In-distribution wins do not count (`CLAUDE.md`).

The three Phase-5 acceptance targets (`CLAUDE.md`):
1. FPR ≤ 1% on fresh phone photos;
2. AUROC ≥ 0.95 after social-media laundering;
3. robustness to transfer attacks (the RAID benchmark).

### Bottom line
**None of the three targets is met.** Our three official hypotheses all failed (H4 REFUTED, H1 FAIL, H2 NOT_SUPPORTED). That triggered the project's stop rule. The owner overrode it for exploration (D-016). The best exploratory detector then failed its own kill test (E7).

**The central obstacle we found is not the generators. It is the real photos.** A detector learns what *our* real photo collections look like. When it meets a new kind of real photo, it calls many of them fake. The laundered difference between real and diffusion images is genuine and survives JPEG and resizing (E5). But every detector we built partly learned the real-photo *source*, so FPR blows up on unseen real pipelines.

| Phase-5 target | Status | Key numbers (source) |
|---|---|---|
| FPR ≤ 1% on fresh phone photos | **UNTESTED** on modern phones: the owner's photos (H3) were never supplied. **NOT MET** on any unseen real source. | Best detector, real source never seen in training, at `jpeg75_resize05`: Open Images 16.0% (32/200), RAISE 19.1% (35/183) (`experiments/016-loso/results/tables.md`). On the old VISION phones (2013–2017 era [exact range differs between sources, see §8]) it reaches 0.6% (6/1074), but 358 reals per group cannot certify ≤ 1% even with zero errors: the upper bound is 1.03% (`experiments/007-worst-env-conformal/official/critique.md` §3). |
| AUROC ≥ 0.95 after laundering | **NOT MET** under the project's rule (unseen conditions). Reached only on real sources the detector was trained on. | In-source: block-8 probe 0.953 [0.939, 0.966] `fov512` / 0.947 [0.933, 0.960] `orig` macro AUROC at `jpeg75_resize05` (`experiments/015-midlayer-probe/results/tables.md`). On a held-out real source: 0.917 (VISION), 0.809 (Open Images), 0.689 (RAISE) (`experiments/016-loso/results/tables.md`). Phase 3's best published or baseline detector: 0.875 (B0) (`experiments/003-baselines-eval/tables.md`). |
| Transfer-attack robustness (RAID) | **UNTESTED** in proper form. The pilot was uninformative. | B0 failed the validity gate on RAID's clean images (AUROC 0.643 [0.560, 0.723] < 0.80). Community Forensics, not part of the attack ensemble, fell from 0.952 to 0.719 at ε = 8/255 (`experiments/010-raid-transfer/pilot/tables.md`). |

### What the project did establish
1. **Evaluation shortcuts are everywhere, and they inflate published-style numbers.** Zoom/scale differences, image texture, caption-twin content leakage, a content-confounded real source (RAISE), and Firefly files that are secretly JPEG (Section 4.1).
2. **Diffusion-specific traces are real but fragile.** The VAE decoder trace is visible on clean images (pair AUROC 0.962 in `fov512`) but a 0.5× downscale erases it (≈ 0.60) (`experiments/011-vae-trace/pilot/tables.md`, critique).
3. **A real-vs-diffusion difference survives laundering** in mid-level DINOv2 features, even with content controlled: twin-paired AUROC 0.927–0.953 (`experiments/014-statistical-field/results/critique.md` F3).
4. **But detectors learn the real source.** On a new real pipeline, polished "enthusiast" photos (bokeh, HDR, telephoto) score as fake (`experiments/016-loso/results/critique.md` F5). Modern phone photos have exactly those properties, which is why the untested H3 matters.
5. **False-positive tails are per device, not per platform.** One 2015 phone (Microsoft Lumia 640) carried 30 of 30 B0 false positives in `fov512` (`experiments/007-worst-env-conformal/official/critique.md` §2).

### Recommended method, in one sentence (a recommendation, not validated)
Scale-normalise every image (`fov512`), read DINOv2-base block-8 features, train a logistic probe on laundering-augmented copies with reals from as many different photo pipelines as possible, and calibrate the threshold on held-out reals *from the exact source and scale it will be used on*, monitoring FPR per source and per device. Expect about 0.95 laundered AUROC on known real sources, and 16–19% FPR on unknown ones (Section 5).

---

## 2. What we built and how we tested

### 2.1 Phases
| Phase | What happened | Main file |
|---|---|---|
| 0 Setup | Intel Mac with an AMD GPU (D-001…D-004), then a GPU PC: WSL2 Ubuntu 22.04, RTX 4060 Ti 8 GB (D-011, D-012) | `DECISIONS.md` |
| 1 Survey | 34 papers + 8 datasets, later 13 diffusion-specific papers; headline numbers re-checked against PDFs (17/17 confirmed) | `docs/01-survey-explainer.md`, `kb/index.md`, `kb/verification-pass-2026-10-24.md` |
| 2 Hypotheses | 8 literature gaps, 7 ranked hypotheses; later 5 diffusion-specific ones (D-014) | `docs/02-hypotheses.md`, `docs/06-diffusion-hypotheses.md`, `kb/ideas/` |
| 3 Baselines | 4 published detectors + our baseline B0 on one fixed evaluation set | `experiments/003-baselines-eval/` |
| 4 Tests | Pilots, then the official H4, H1, H2; then exploratory E5–E7 under D-016 | `docs/04-phase4-plan.md` (Amendments 1–16), `experiments/005…016` |
| 5 Report | This file | — |

### 2.2 Evaluation protocol (D-009, with later additions)
- **Preprocessing:** EXIF orientation applied, metadata stripped, a 512-px centre crop saved as PNG. Identical for both classes. (`DECISIONS.md` D-009)
- **Conditions:** `clean`, `jpeg75_resize05` (main), `social_sim`. Phase 4 added `screenshot_sim` and `mavali_medium` as held-out launderings never used for fitting, plus factor ablations (`docs/04-phase4-plan.md`, shared infrastructure).
- **Metrics:** AUROC, AP, balanced accuracy, FPR@95%TPR, TPR@1%FPR, with bootstrap 95% CIs. From Amendment 3 on, CIs resample whole devices (VISION) or caption families, not single images.
- **Breakdowns:** per generator; leave-one-generator-out; per real pool (VISION native / WhatsApp / Facebook-HQ / RAISE).
- **Splits:** VISION by device (10 train, 10 calibration, 25 test phones; 99 / 43 / 358 native photos); WhatsApp/Facebook copies follow their original's device. FLUX is test-only. (`docs/04-phase4-plan.md`, reconciliation table)
- **Scale arms (D-013, still marked "PROPOSED" in `DECISIONS.md`):** every Phase-4 result is reported in `orig` and `fov512`.

### 2.3 Guards against fooling ourselves
- **Pre-registration.** Each test's pass/fail rule was committed before its result. Changes after a result are dated amendments, and post-result choices are disclosed (e.g. Amendment 4 adopted H2-v2's rule after seeing pilot labels).
- **One independent critic per result** (opus/high). Critics re-computed numbers from raw scores, looked for leakage, and corrected several readings (e.g. "23.4% cross-zoom FPR" became "FPR inflated ≥ 5×", `experiments/008-scale-sensitivity/pilot/critique.md`).
- **Pre-run code reviews.** These caught real bugs before any result. Example: E3's clean control differed from RAID's own crop by 64–83 grey levels, and 52 of 100 RAID reals had black padding borders (`experiments/010-raid-transfer/code_review.md`, via `docs/05-phase4-pilot-synthesis.md` §3.6).
- **D-008 numeric guard.** Every GPU feature pass was compared with CPU on a fixed 32-image subset and aborted if max |Δlogit| > 1e-3. This guard caught two silent-error incidents on the Mac (D-008, D-010) and rejected the SDXL VAE on the GPU (`LOG.md` 2026-10-24 08:55).
- **Validity gates.** A trivial texture baseline (image sharpness, the Laplacian) next to every score; a headroom rule (H1); a clean-AUROC gate (E3); a mechanism gate per arm (from the H-D1 critique on).
- **Robustness of labels.** From Amendment 13 on, each label is re-computed with Firefly removed and with each generator dropped in turn, plus a "twin-disjoint" refit that removes caption-twin leakage.
- **Stop-rule counting (Amendment 13).** Only official FAIL / REFUTED / NOT_SUPPORTED labels count. UNINFORMATIVE labels mean "not tested".

### 2.4 Data
| Set | Count | Licence | Role |
|---|---|---|---|
| VISION (Shullani et al. 2017) | 500 native phone photos from 35 devices, + 500 WhatsApp + 500 Facebook-HQ copies | CC BY-SA 4.0 | Main real set; platform pairs |
| RAISE via B-Free "extended Synthbuster" | 400 resized DSLR PNGs | NON-COMMERCIAL | Real set at the fakes' scale; the caption source of all fakes |
| Synthbuster | 120 each: DALL-E 2, DALL-E 3, Firefly, Midjourney v5, SDXL, SD2 | CC BY-NC-SA 4.0 | Fakes |
| B-Free extended | 150 SD3.5-large, 150 FLUX (test-only), 300 latent-diffusion (excluded: 256 px) | NON-COMMERCIAL | Fakes |
| RAID subset | 1,500 files (500 ids × clean, ε 8, ε 16) | MIT (upstream image licences [unverified]) | E3 transfer-attack pilot |
| Open Images V7 validation | 600 photos | CC BY 2.0 per image | E7 third real source |

Sources: `DECISIONS.md` D-005; `LOG.md` 2026-10-21 23:21, 2026-10-24 01:30, 2026-10-25 20:56; `env/SOURCES.md`. After excluding latent-diffusion and one duplicate, the evaluation table has 2,919 rows (`LOG.md` 2026-10-21 23:42).

**Licence consequence:** most fakes and the RAISE reals are non-commercial. Everything here is research-only.

### 2.5 Why pilot numbers differ from official numbers
- From 2026-10-23 ~09:40 to 2026-10-24 19:47, `grip.unina.it` was down. It hosts RAISE, SD3.5, FLUX and the B-Free weights (`LOG.md`).
- The owner chose "pilot now, official later". Pilots used VISION + 6 Synthbuster generators only: no RAISE, no SD3.5, no FLUX.
- Pilot AUROCs sat near 0.98–0.99. The critics showed these were ceiling effects driven by zoom, texture and content shortcuts (`docs/05-phase4-pilot-synthesis.md` §1). On the full official data, the best laundered baseline was 0.875 (`experiments/003-baselines-eval/tables.md`).
- Per Amendment 2, no pilot counts toward the stop rule.
- When grip returned, all 3,220 images re-verified (24 OK, 0 mismatches), and Phase 3 was reproduced on the GPU PC: max |ΔAUROC| 0.0019 over 136 cells (`LOG.md` 2026-10-24 20:00 and 2026-10-25 08:32; `experiments/003-baselines-eval/step6_gpu_tables.md`).

---

## 3. Results

### 3.0 Starting point: the Phase 3 baseline table
`experiments/003-baselines-eval/tables.md` (Mac run, `orig` arm; reals = VISION native + RAISE test):

| Detector | AUROC clean | AUROC `jpeg75_resize05` | FLUX AUROC clean / laundered | TPR@1%FPR laundered |
|---|---|---|---|---|
| Community Forensics | **0.992** | 0.800 | 0.984 / 0.593 | 0.321 |
| B-Free | 0.970 | 0.714 | 0.913 / 0.469 | 0.198 |
| RINE | 0.844 | 0.807 | 0.816 / 0.689 | 0.132 |
| ClipDet (CLIP-only variant) | 0.780 | 0.595 | 0.649 / 0.460 | 0.019 |
| B0 (DINOv2 last block + probe) | 0.932 | **0.875** | 0.886 / 0.813 | 0.176 |

- No method reaches 0.95 laundered.
- At the published default threshold, B-Free flags 17.3% of VISION phone photos as fake on clean images (`experiments/003-baselines-eval/notes.md` finding 4).

### 3.1 Official hypothesis tests (these count)

**H4 — Does a training-free "drift" score only work because reals are JPEG and fakes are PNG?** → **REFUTED** (failed hypothesis 1 of 3).
- *Question.* A RIGID-style score measures how much DINOv2 features move under small noise. Phase 2 predicted that its accuracy depends on a JPEG-real / PNG-fake format shortcut.
- *Answer.* No. Giving the RAISE reals JPEG history did not help the score: the gap AUROC(M) − AUROC(ii) is −0.042 [−0.056, −0.028] in the label cell (σ = 0.02, `orig`, with Firefly), far below the +0.10 predicted. The validity gate passed (AUROC(ii) 0.619 [0.572, 0.664] vs texture 0.538). `fov512` is also REFUTED (gap −0.012) (`experiments/006-format-shortcut/official/tables.md`).
- *Critic.* The label stands. The narrow wording is "no real-side JPEG shortcut ≥ 0.03 at 512-crop scales"; the score is still format-sensitive by 0.02–0.06, with a sign that depends on scale. The gate passes only because of Firefly. The score is weak anyway: TPR@1%FPR ≤ 0.037 in every arm. The critic also found **caption-twin leakage** in the Phase 3 splits (Section 4.1) (`experiments/006-format-shortcut/official/critique.md` F1–F3).
- *Side result.* The "B0-JM" fix (JPEG-compress only the training fakes) failed: VISION − RAISE gap 0.088 instead of ≤ 0.03. It also hurt the WhatsApp/Facebook pools by 0.017–0.033 (critique F5).

**H1 — Does removing "laundering directions" from the features help?** → **FAIL** in all 4 runs (failed hypothesis 2 of 3).
- *Question.* We measured how real photos' features move under WhatsApp, Facebook and JPEG+resize, then projected those directions out before the probe (arm A2). Prediction: ≥ +0.03 laundered AUROC on FLUX and on the LOGO macro, versus augmentation alone (A1).
- *Answer.* It made things worse. In `orig`: A2 − A1 = −0.0149 [−0.023, −0.007] macro and −0.0319 [−0.045, −0.020] on FLUX, with a −0.020 clean loss. In `fov512`: +0.0019, far below +0.03. The result holds for every k and every dropped generator (`experiments/005-nuisance-projection/official/gate_robustness.md`, `critique.md` §1).
- *Critic.* FAIL is right. The one criterion met (threshold drift −67%) is not a win: it came from raising every FPR (test natives 8.1% → 16.5% at a 5% threshold). Lesson: in DINOv2's last layer, "real-photo nuisance" and "fake signal" share directions (`critique.md` §2).

**H2 — Does a "worst platform" threshold keep FPR low under platform shift?** → **NOT_SUPPORTED** for the primary model A0, and for B0, CommFor, RINE and B-Free (failed hypothesis 3 of 3).
- *Question.* Instead of one pooled 1%-FPR threshold, take the strictest threshold needed by any platform group (native, WhatsApp, Facebook, plus synthetic platforms).
- *Answer.* At matched detection rate it was no better. At α = 1% it was significantly worse for A0: +0.25 pp FPR [+0.01, +0.60] (`experiments/007-worst-env-conformal/official/critique.md` §2). Plain pooled calibration at nominal 1% gave B0 1.28% (23/1790) pooled and 2.51% (9/358) on WhatsApp (`summary_orig.md`, reproduced in the critique).
- *Critic.* The label stands, with one disclosure: α = 0.5% is a built-in tie that always votes NOT_SUPPORTED; without it the majority still holds. **The key finding: the FPR tail is one phone, not one platform.** Device D17 (Microsoft Lumia 640) carried 17 of 23 B0 false positives in `orig` and 30 of 30 in `fov512`, with 40% of its photos flagged on every platform. 21–24 of the 25 phones had 0% (`critique.md` §2).
- *Cost of strictness.* Pushing every platform to about 1% needs pooled α ≈ 0.25%. B0 then catches 21.7% of clean fakes, 13.7% of laundered fakes and **5.3% of unseen FLUX** fakes (`critique.md` §3).

**Stop rule.** H4 + H1 + H2 = 3 failed hypotheses, which triggers CLAUDE.md's stop rule. The owner explicitly asked to continue exploring (D-016, 2026-10-25 20:19). All later work is **exploratory** and still pre-registered and critiqued.

### 3.2 Exploratory results (pilots and D-016 experiments; none counts as a hypothesis test)

**E1 — Are published detectors sensitive to zoom?** (`experiments/008-scale-sensitivity/pilot/`)
- *Answer.* Yes. Changing only the crop rule (`orig` vs `fov512`) moves laundered AUROC by +0.062 (CommFor), +0.228 (RINE) and −0.217 (ClipDet). A threshold set at one scale inflates FPR at least 5× at the other (`LOG.md` 2026-10-24 05:20; `pilot/critique.md`).
- *Critic.* Sound. "B0 is not sensitive" holds only if B0 is retrained per arm. A fixed `orig`-trained B0 drops 0.981 → 0.923 laundered at `fov512`; a fixed `fov512`-trained B0 is stable, but its TPR@1%FPR is lower (0.683 vs 0.900) (`docs/05-phase4-pilot-synthesis.md` §2; `pilot/tables.md`).

**E1b — Is it zoom or resampling?** (`experiments/008-scale-sensitivity/pilot_du2/`)
- *Answer.* Confounded. The down-then-up arm blurs phone photos (≈ 4.8× downsampling) far more than fakes (≈ 2×). Only one label survives a matched control: the fixed `orig`-trained B0's laundered loss is caused by zoom. The critic ruled that no `du2` number may be quoted (`pilot_du2/critique.md`).

**E2 #6 — A probe trained only on reals vs their own SD-VAE round-trips** (`experiments/009-vae-aligned/pilot/`)
- *Answer.* INCONCLUSIVE: 0.819 [0.769, 0.861] clean, 0.778 [0.731, 0.817] laundered on 6 generators it never saw (`pilot/tables.md`).
- *Critic.* Not a VAE-trace detector. It separates unseen-device photos from their *own* round-trips at only 0.66 clean / 0.56 laundered, and scores RAID web reals at 0.52. It rides a "not straight from a camera" axis. In `fov512` its laundered AUROC falls to 0.681 (`pilot/critique.md` F1). One positive: at VISION's 1% threshold it flags 2.1% of RAID reals, against 67% for B0 (`docs/05` §2).

**E2 #7 — VAE double-reconstruction ratio** → FAIL: 0.601 clean, below the texture baseline's 0.778; 0.318 laundered. Dropped (`experiments/009-vae-aligned/pilot/tables.md`; `docs/05` §2).

**E3 — Transfer attacks (RAID) and "canonicalisation"** (`experiments/010-raid-transfer/pilot/`)
- *Answer.* UNINFORMATIVE. B0's clean AUROC on RAID was 0.643 [0.560, 0.723], under the 0.80 gate. Community Forensics, outside the attack ensemble, fell 0.952 → 0.719 at ε = 8/255 (`pilot/tables.md`).
- *Critic.* B0 calls 77% of RAID reals fake: a "camera photo = real" prior learned from VISION. Test-time JPEG + resize nearly cancels the attack, but costs about as much clean AUROC (0.08–0.16) (`pilot/critique.md`). The attack question was then paused by the owner (D-014).

**E4 — How visible is the SD-VAE decoder trace?** (`experiments/011-vae-trace/pilot/`)
- *Answer.* Real but fragile. A probe separates a photo from its own SD-VAE round-trip at 0.862 [0.833, 0.894] (`orig`) and 0.962 [0.953, 0.972] (`fov512`) clean, but only ≈ 0.60 after `jpeg75_resize05` (`pilot/tables.md`; `LOG.md` 2026-10-24 13:55).
- *Critic.* The 0.5× resize, more than JPEG 75, erases the trace. The 0.99 within-pair win rate is reproduced by a plain blur, so it is not VAE-specific (`pilot/critique.md`).

**H-D1 — Self-referenced trace contrast** (score an image by how it moves under one more VAE round-trip) (`experiments/012-self-ref/pilot/`)
- *Answer.* At chance: 0.533 clean / 0.496 laundered / 0.539 `social_sim`. A blur control was equal or better (`pilot/tables.md`).
- *Critic.* Substantively a FAIL. In `orig` the mechanism exists on clean images (0.704; SD-family 0.867) but vanishes under laundering (0.484). Three reconstruction / fixed-point scores have now failed on laundered inputs. Stop spending effort on them (`pilot/critique.md`).
- *Side number.* E4's pair-trained probe, which never saw a fake, scores 0.892 clean / 0.665 laundered on 6 unseen generators in `fov512` (`pilot/tables.md`).

**H-D2 — Does the 8-px latent grid survive non-dyadic resizing?** (`experiments/013-latent-grid/pilot/`)
- *Answer.* UNINFORMATIVE: the validity gate failed (0.000 of SD fakes above the reals' 99th percentile); scores at chance (0.49–0.53) (`LOG.md` 2026-10-24 19:30).
- *Critic.* An 8-px grid *does* exist in about 70% of SD fakes, but VISION's camera-JPEG grid sits on the same 8-px lattice with equal strength. The "dyadic collision" mechanism is not supported: the trace decays smoothly with downscaling (`pilot/critique.md`).

**E5 — Is there any statistical difference left after laundering?** (`experiments/014-statistical-field/results/`)
- *Answer.* Yes. P1 CONFIRMED: content-matched pairs are separable on clean images (hand-built statistics, C2ST 0.777 [0.754, 0.800]). P2 REFUTED: laundering does *not* close the gap (DINOv2 C2ST 0.940; MMD significant for 8/8 generators). P3 REFUTED. P4 CONFIRMED: most of the hand-built statistics' separation is scale and format (0.972 vs 0.723) (`results/tables.md`).
- *Critic.* The 0.940 headline is inflated by laundered training and by VISION content. The defensible content-controlled laundered numbers (RAISE real vs its own caption twins): block 8 **0.927** (`orig`) / **0.953** (`fov512`); B0 0.837 / 0.869; hand-built statistics 0.687 / 0.722. **Caveat:** one real source only (RAISE), so a "diffusion signature" cannot yet be separated from "RAISE's processing" (`results/critique.md` F1–F3).

**E6 — A mid-level (block-8) DINOv2 detector** (`experiments/015-midlayer-probe/results/`)
- *Answer.* P1 PASS: trained on laundered copies, block 8 reaches 0.947 laundered macro vs 0.855 for a matched B0 (+0.092 [+0.069, +0.121]), FLUX 0.874 vs 0.797, clean 0.970; held-out launderings 0.961 (`screenshot_sim`) and 0.942 (`mavali_medium`). **P2 FAIL:** trained on one real source and tested on the other, AUROC is 0.644 (RAISE→VISION) and 0.604 (VISION→RAISE), against 0.85 required. **P3 FAIL:** FPR at the pooled 1% threshold is native 0.8%, WhatsApp 3.1%, Facebook 2.2%, RAISE 7.1% (`results/tables.md`).
- *Critic.* No leak; labels stand. The failure is **real photos flagged as fake**, not fakes missed: a RAISE-only probe flags 41.9% of VISION natives at `jpeg75_resize05` (`orig`), 12.8% in `fov512`. More images from the same source do not help; more *kinds* of source might. The FLUX part of P1 is fragile on a larger training set (+0.040 [−0.007, +0.089]). `fov512` is much better cross-source (0.826 vs 0.644) (`results/critique.md` F1–F4). The critic recommended exactly one more bounded test, E7. The lead raised the stop-rule limit by one for it (D-016).

**E7 — Leave-one-real-source-out: does training on several real sources transfer to an unseen one?** (`experiments/016-loso/results/`)
- *Answer.* **KILL** under Amendment 16, in both threshold readings, all 3 arms and all 10 drop-one-generator subsets. `fov512`, `jpeg75_resize05`, block 8, trained on the other two sources:

  | Held-out real source | AUROC | FPR at the transferred 1% threshold |
  |---|---|---|
  | VISION | 0.917 [0.893, 0.938] | 0.6% (6/1074) |
  | RAISE | 0.689 [0.650, 0.726] | 19.1% (35/183) |
  | Open Images | 0.809 [0.774, 0.843] | 16.0% (32/200), CP 95% [11.2%, 21.8%] |

  (`results/tables.md`). Block 8 beats the last block by +0.057 [+0.035, +0.079], and a second training source adds +0.032 [+0.025, +0.039] AUROC. Both secondary criteria are met, but the per-source bar fails everywhere.
- *Critic.* KILL is correct; no bug. The RAISE fold is partly unfair by construction (all fakes use RAISE captions), but Open Images alone triggers the kill: its FPR CI lies wholly above 10%. Every threshold available without knowing the image's source gives ≥ 16% on Open Images. The flagged photos are polished enthusiast shots (bokeh, telephoto wildlife, HDR, saturated colour). Grayscale, EXIF, JPEG quality and scale are ruled out; the style reading comes from about 30 images and is suggestive, not controlled. Adding RAISE to training *raised* Open Images FPR from 10.5% to 16.0% (`results/critique.md` F2–F6). The mid-level line is closed (D-016).

### 3.3 Summary of labels
| Experiment | Kind | Pre-registered label | Critic's reading |
|---|---|---|---|
| H4 drift format shortcut (006) | official | REFUTED | stands; narrow wording |
| H1 nuisance projection (005) | official | FAIL | stands; laundering and fake signal entangled |
| H2 worst-env threshold (007) | official | NOT_SUPPORTED | stands; FPR tail is per device |
| E1 zoom (008) | pilot | scale-sensitive | sound; reframed |
| E1b zoom vs resampling (008/du2) | pilot | — | confounded; only the B0 label survives |
| E2 #6 aligned VAE (009) | pilot | INCONCLUSIVE | not a trace detector |
| E2 #7 VAE ratio (009) | pilot | FAIL | correct; dropped |
| E3 RAID (010) | pilot | UNINFORMATIVE | correct |
| E4 VAE trace (011) | diagnostic | not detectable (`orig`); not laundering-robust | sound |
| H-D1 self-reference (012) | pilot | UNINFORMATIVE | substantively FAIL |
| H-D2 latent grid (013) | pilot | UNINFORMATIVE | mechanism not supported |
| E5 statistical field (014) | exploratory | P1, P4 confirmed; P2, P3 refuted | sound; single real source |
| E6 mid-level detector (015) | exploratory | P1 PASS; P2, P3 FAIL | sound; source-specific |
| E7 leave-one-source-out (016) | exploratory | KILL | correct; line closed |

---

## 4. What the project established (cross-cutting findings)

### 4.1 Evaluation data is full of shortcuts
A detector can score well for the wrong reason. Each of these was measured on our data:
- **Zoom / scale.** Median short side: VISION natives 2448 px vs fakes 768–1024 px (Firefly 1792) (`experiments/001-eval-set/manifest_raw.csv`, via `LOG.md` 2026-10-23 18:12). An `orig` crop shows phone photos 2.5–3× more zoomed in. In `fov512`, B0's gaps between real pools disappear (native / WA / FBH: 0.992 / 0.966 / 0.982 → 0.979 / 0.979 / 0.980), and threshold drift falls from 0.197 to 0.004 (`docs/05-phase4-pilot-synthesis.md` §2–3).
- **Texture.** Image sharpness alone separates the classes in `orig` (Laplacian AUROC 0.778). It explains the inverted RIGID score (0.292 in the pilot) and the VAE-error scores (`docs/05` §3.2; `experiments/006-format-shortcut/pilot/critique.md`).
- **Caption-twin leakage.** The Phase 3 splits did not group fakes with the RAISE photo whose caption made them. B0 scores RAISE test reals 0.786 when the twin is in both train and test, versus 0.935 with no twin (n = 18). That leakage, not format or zoom, explains most of the Phase 3 "VISION 0.962 vs RAISE 0.876" gap (`experiments/006-format-shortcut/official/critique.md` F1). Amendment 13 added twin-disjoint refits to all later runs.
- **RAISE content confound.** Every fake was prompted from a RAISE caption. So a probe trained on VISION reals can separate the classes partly by content, and any test with RAISE held out is unfair by construction (`experiments/015-midlayer-probe/results/critique.md` F1; `experiments/016-loso/results/critique.md` F3).
- **Firefly JPEG.** Synthbuster's Firefly files are JPEG q≈95 saved under `.png` names. Format must be read from content, not file extension (`DECISIONS.md` D-009; `kb/datasets/synthbuster.md`). Firefly alone decided H4's validity gate (`006/official/critique.md` F2).
- **Platform copies.** VISION WhatsApp copies are separated from fakes at 0.98–0.99 by DINOv2 even when scale-matched, i.e. mostly by content. They are not evidence of a diffusion signal (`experiments/014-statistical-field/results/critique.md` F2).

### 4.2 Diffusion-specific traces exist but downscaling erases them
- The SD-VAE trace is visible in mid-level features on clean images (0.962 `fov512`), and drops to about 0.60 after `jpeg75_resize05` (`experiments/011-vae-trace/pilot/tables.md`).
- The resize does most of the damage. JPEG only: about 0.78–0.80; resize only: about 0.64–0.67 (`011/pilot/critique.md`, via `LOG.md` 2026-10-24 14:40).
- Reconstruction and fixed-point scores (VAE ratio, self-reference) failed on laundered inputs three times (`012/pilot/critique.md`).
- The latent 8-px grid is present, but camera JPEG produces the same lattice (`013/pilot/critique.md`).
- **Implication:** a detector built purely on decoder artefacts will not survive social-media downscaling.

### 4.3 The real-vs-diffusion difference survives laundering, but detectors learn the real source
This is the project's central result.
- **The difference is real.** With content, scale and format controlled, block-8 features still separate RAISE photos from their own caption-twin fakes at 0.927–0.953 after laundering (`014/results/critique.md` F3).
- **The detector learns the real source.** In E6 and E7, every probe trained on some real sources flagged many photos from an unseen source: 41.9% of VISION natives for a RAISE-only probe in `orig` (`015/results/critique.md` F1); 16.0% of Open Images and 19.1% of RAISE with two training sources (`016/results/tables.md`).
- **Mechanism (suggestive).** The flagged Open Images photos are polished, "stock-like" shots; the lowest-scoring ones are casual snapshots (`016/results/critique.md` F5). The probe's "real" direction is partly "casual snapshot".
- **Why it matters for Phase 5.** Modern phones produce computational bokeh, HDR and heavy processing. Those are exactly the properties that raised FPR here. This is why the owner-photo test (H3) is the deciding experiment, and why a pooled ≤ 1% claim from old phones does not transfer [prediction, untested].
- **Diversity helps AUROC but not enough FPR.** Each added source bought about +0.03 AUROC (`016/results/tables.md`), but adding RAISE raised Open Images FPR (10.5% → 16.0%) (`016/results/critique.md` F6).

### 4.4 FPR tails are per device, not per platform
- One Lumia 640 carried essentially all of B0's false positives (Section 3.1, H2). A "1% on average" can mean "0% for most phones, 40% for some" (`007/official/critique.md` §3).
- For the block-8 probe in `orig`, false positives were instead spread across about 15 devices, mostly WhatsApp/Facebook copies (`015/results/critique.md` F4). Different features, different tails.
- **Report FPR per device and per source, never only pooled.**

### 4.5 Thresholds do not transfer across scale
- Ranking (AUROC) often survives a change of input scale; thresholds do not. Cross-zoom FPR inflation was at least 5× (`008/pilot/critique.md`). RINE laundered `orig`→`fov512` reached 31.6% FPR (`LOG.md` 2026-10-24 05:50).
- A threshold is valid only for the scale, channel and source it was calibrated on.

### 4.6 Our data cannot certify ≤ 1% FPR
- With 358 reals per platform group, even 0 errors gives a Clopper-Pearson upper bound of 1.03% (`007/official/critique.md` §3).
- Copies of the same photo (native, WhatsApp, Facebook) are not independent, and errors cluster by device. Device-bootstrap CIs are the honest ones: A0 pooled@1% = 1.40% [0.06%, 3.77%] (`007/official/critique.md` §1).
- (derived) For a two-sided 95% Clopper-Pearson upper bound below 1% with zero errors, about 368 independent reals per group are needed. One or more errors needs more.

---

## 5. Recommended method (A RECOMMENDATION, NOT A VALIDATED METHOD)

**Read this first.** This configuration failed its own pre-registered kill test (E7). It is the best-supported *configuration* in our evidence, not a method that meets the targets. Its mechanism is also not new: frozen multi-layer foundation-model probes are a crowded area (`docs/02-hypotheses.md` §2, novelty check). Our contribution is the evaluation evidence about *where it fails*.

### 5.1 The configuration
1. **Scale normalisation (`fov512`).** Strip metadata, resize the short side to 512 px (antialiased bicubic), centre-crop 512, then take the 224-px centre crop that DINOv2 reads (004 pipeline, `docs/04-phase4-plan.md`).
   - *Evidence:* better cross-source transfer (0.826 vs 0.644, `015/results/tables.md`); no WhatsApp/Facebook recompression effect (`015/results/critique.md` F4); E7 is worse in `orig` (Open Images FPR 35.0% vs 16.0%, `016/notes.md`).
2. **Features:** DINOv2-base, block 8, CLS token + mean patch token (1,536 numbers).
   - *Evidence:* +0.057 [+0.035, +0.079] over the last block on held-out sources (`016/results/tables.md`).
3. **Classifier:** standardise, then a balanced logistic regression; regularisation chosen by grouped cross-validation (groups = device / caption family / image) (`015-midlayer-probe/config.json`, `run.py`).
4. **Training data:** clean + `jpeg75_resize05` + `social_sim` copies of every training image ("T-aug").
   - *Evidence:* laundering augmentation adds +0.046 macro and +0.097 on FLUX for block 8 (`015/results/critique.md` F2).
5. **Reals from as many pipelines as possible:** VISION + RAISE + Open Images at minimum, plus the deployment's own source, with diverse photographic styles (casual and polished).
   - *Evidence:* the learning curve (+0.032 per added source, `016/results/tables.md`).
   - **Caveat:** this exact three-source model was never tested on a fourth, unseen source. E7 only trained on two of three.
6. **Calibration per source and scale:** set the threshold from held-out reals of the *same* source, at the *same* scale and laundering condition as deployment.
   - *Evidence:* a source-specific threshold gave RAISE 2.7% and natives 0.0% in E6 (`015/results/critique.md` F4). A transferred one gave 16–19% (E7).

### 5.2 Measured performance (best available numbers; each with a scope warning)
| Setting | AUROC `jpeg75_resize05` | FPR at the 1% calibration threshold | TPR@1%FPR | Source |
|---|---|---|---|---|
| Known real sources (VISION + RAISE), unseen generators (LOGO), `fov512` | 0.953 [0.939, 0.966] (B0: 0.885) | native 0.6%, WA 0.8%, FBH 0.6%, RAISE 2.7%, all 1.0% (12/1257) | 0.543 (B0: 0.215) | `015/results/tables.md` |
| Same, FLUX only (never seen) | 0.913 [0.880, 0.939] | — | — | `015/results/tables.md` |
| Same, twin-paired RAISE (content controlled) | 0.906 [0.880, 0.929] | — | — | `015/results/tables.md` |
| Unseen real source: VISION held out | 0.917 [0.893, 0.938] | 0.6% (6/1074) | 0.457 | `016/results/tables.md` |
| Unseen real source: Open Images held out | 0.809 [0.774, 0.843] | **16.0%** (32/200) | 0.760 | `016/results/tables.md` |
| Unseen real source: RAISE held out (content-confounded) | 0.689 [0.650, 0.726] | **19.1%** (35/183) | 0.567 | `016/results/tables.md` |

**A more conservative variant.** The same recipe with last-block features (B0 features, `F_last`) gives lower FPR on unseen sources, at far lower detection. Open Images: 4.5% FPR, TPR 0.301. RAISE: 10.4% FPR, TPR 0.261 (`016/results/tables.md`). Neither variant meets ≤ 1% on an unseen source.

### 5.3 Known failure modes
1. **Unseen real pipelines.** 16–19% FPR on new real sources; polished photos look "AI" (Section 4.3).
2. **Modern phones untested.** Computational photography (bokeh, HDR, night modes) is the likeliest failure, and it has not been measured (H3).
3. **Scale changes.** Thresholds break across input scales (≥ 5× FPR inflation, Section 4.5).
4. **Per-device tails.** A single phone model can dominate false positives (Section 4.4).
5. **Unseen-generator gain is fragile.** The FLUX improvement over B0 loses significance on a larger training set (+0.040 [−0.007, +0.089], `015/results/critique.md` F3).
6. **Transfer attacks untested.** Published detectors outside the attack ensemble still lost about 0.23 AUROC on RAID (`010/pilot/tables.md`).
7. **Content confound.** All training fakes were prompted from RAISE captions, so part of what the probe learned may be content.
8. **Research-only licences** for most training data (Section 2.4).

### 5.4 How to deploy it honestly (if at all)
- **Use it as a triage score, not a verdict.** At our best measured operating point it still misses about half of laundered fakes (TPR 0.543).
- **Normalise scale before scoring**, and never reuse a threshold across scales.
- **Calibrate per source.** Collect held-out real photos from each source the detector will see (for example, the owner's phone). Set the threshold there, under the laundering condition expected in deployment.
- **Report FPR with a Clopper-Pearson interval and per device / per source.** Do not claim ≤ 1% unless the interval's upper end is below 1% (about 368+ independent reals with zero errors, derived in §4.6).
- **Monitor drift.** Track the score distribution of incoming real images per source. A shift toward "fake", like Open Images' median moving from −6.06 (VISION calibration) to −0.72 (`016/results/critique.md` F4), means the threshold no longer holds.
- **Unknown sources should get no confident verdict.** We did not build or test a detector for "this real source is new"; this is a recommendation [untested].

---

## 6. Open problems and next steps (ranked)

1. **Owner phone photos as the acceptance test (H3).** This is the single most valuable next step (`016/results/critique.md` Q4). It needs no download, the licence is clean, and it is the Phase-5 target population.
   - Design (from `kb/ideas/neural-isp-fpr-audit.md` and the E6 critic): at least 150–250 unedited originals from 2021+ phones with neural ISPs, covering default, night, portrait and ≥ 5× zoom modes, plus WhatsApp-sent copies. Photos stay local and are never committed.
   - Score the **frozen** Section 5 configuration and B0, pre-registered, before looking. Report FPR per capture mode and per phone.
   - Power: 250 photos with 0 errors cannot certify ≤ 1% (§4.6). Treat it as a screen for large failures, which is how `kb/ideas/neural-isp-fpr-audit.md` pre-registers it.
2. **A multi-source real training set with style diversity.** E7 shows that 2 sources are far from enough, and that adding a source can *raise* FPR on another.
   - Needed: many real pipelines (modern phones, DSLRs, Flickr-style polished photos, screenshots), balanced counts, and fakes whose content matches those reals (caption twins for new reals need generator weights over 5 GB, which needs owner approval under D-015).
   - Test with leave-one-source-out and the E7 kill rule unchanged.
3. **What to test before any claim.**
   - Freeze the detector (features, training data, threshold rule, seeds) in a pre-registration.
   - Test on a real source never seen in training or calibration, with FPR per device and a Clopper-Pearson bound.
   - Show that trivial probes fail on the new pairs: texture, file size, dimensions, blur (`014/results/critique.md` recommendation 4).
   - Check that the score does not track the RAISE resize kernel (same source, recommendation 4e).
   - Run the proper transfer-attack test: an in-domain probe, a canonicalised-input arm, noise controls and ≥ 300 unpadded test reals. This needs about 0.9 GB of RAID originals, an owner decision (`010/pilot/critique.md`; `docs/05` §5c).
4. **Owner decisions still open** (`README.md`): D-013 (`fov512` primary; all evidence favours it); a larger backbone (DINOv2-L / DINOv3, gated licence); RAID originals.
5. **Housekeeping owed by the lead (not done here, because this task was read-only outside this file):**
   - The KB idea files for H1, H2 and H4 still say "proposed" or "official run pending" (`kb/ideas/laundering-nuisance-projection.md`, `worst-env-conformal-fpr.md`, `drift-format-shortcut.md`).
   - No KB entries yet record the E5–E7 results as negative results, which CLAUDE.md requires.
   - `README.md` "Current status" still describes the 2026-10-24 blocker.
   - `env/STORAGE.md` says about 8.8 GB of cumulative downloads; `LOG.md` 2026-10-25 20:56 says about 9.9 GB.

---

## 7. Reproducibility

### 7.1 Environment
| Item | Value | Source |
|---|---|---|
| GPU PC | WSL2 Ubuntu 22.04, RTX 4060 Ti 8 GB, driver 591.86 | `LOG.md` 2026-10-23 09:49; D-012 |
| Python | CPython 3.12.15, uv-managed in `env/.python` (uv 0.13.4 from PyPI) | D-012; `env/SOURCES.md` |
| Packages | torch 2.8.0+cu128 / torchvision 0.23.0 on Linux; torch 2.2.2 on the Intel Mac; numpy < 2; all pinned in `env/uv.lock` | D-011 |
| Numeric guard | D-008: CUDA vs CPU, TF32 off, abort if max \|Δlogit\| > 1e-3 | D-008, D-011 |

Setup commands are in `env/SETUP_WSL.md` steps 4–6:
```
cd env && python3.12 -m venv .bootstrap && .bootstrap/bin/pip install uv==0.13.4
UV_PROJECT_ENVIRONMENT=.venv .bootstrap/bin/uv sync --frozen --python python3.12 && cd ..
env/.venv/bin/python -I env/bootstrap_data.py            # re-download, SHA256-verified
env/.venv/bin/python -I env/bootstrap_data.py --verify-only
env/.venv/bin/python -I experiments/lib/tests/run_all.py
```

### 7.2 Data provenance
- `env/SOURCES.md`: URL, version or commit, licence and SHA256 for every dataset, weight file and repository.
- `env/STORAGE.md`: large items, why each is kept, and deletions (the confounded `du2` crops, D-015).
- `experiments/001-eval-set/manifest_raw.csv`: the evaluation images, checked by `env/bootstrap_data.py`. The Open Images subset is listed in `experiments/016-loso/openimages_manifest.csv`.
- `data/` and `third_party/` are git-ignored and re-created by `env/bootstrap_data.py`.

### 7.3 Commands for the main results
| Result | Command (from the repo root) | Source |
|---|---|---|
| Phase 3 baselines | `env/.venv/bin/python -I experiments/003-baselines-eval/run.py prep` → `score` → `b0` → `report` | `env/SETUP_WSL.md` step 6 |
| Phase 4 features and official H4 | the sequence in `experiments/006-format-shortcut/notes.md` ("Exact command sequence"), ending in `run_official.py` | that file |
| Official H1 | `experiments/005-nuisance-projection/official/launch.sh` | that file |
| Official H2 A-arms | `analyze.py --v2 --scores 005/official/orig/scores_long.csv.gz --splits 007/official/splits_orig.csv --groupings platform --min-group 100 --label OFFICIAL` | `LOG.md` 2026-10-25 20:27 |
| E5 | `env/.venv/bin/python experiments/014-statistical-field/run.py all` | `014/notes.md` |
| E6 | `env/.venv/bin/python -I experiments/015-midlayer-probe/run.py all --workers 8` | `015/notes.md` |
| E7 | `env/.venv/bin/python -I experiments/016-loso/run.py all --workers 14` | `016/notes.md` |

### 7.4 Commit hashes (from `git log`)
| Commit | Content |
|---|---|
| `286c602` | Phase 0 |
| `c4242d0` | Phase 1 complete |
| `389d308` | Phase 2 complete |
| `474ce86` | Phase 3 complete (baseline table) |
| `e1d7896`, `ab63196` | Phase 4 pilots (H1, H2), Amendments 2–4 |
| `d9e2f38` | H4 pilot |
| `a1b19c5`, `84640a1` | E1 pilot; E1b critique |
| `60c6e43`, `560de88` | E2 pilot; critique |
| `74ee53f`, `428a066` | E3 pilot; critique |
| `835dab6`, `141853b` | E4 pilot; critique and Amendment 12 |
| `a81cfed`, `c4b0a2d` | H-D1 pilot; critique |
| `4c60cdd`, `d7f27e9` | H-D2 pilot; critique |
| `c8eeeba` | Phase 3 reproduced on the GPU PC |
| `b5e3aca`, `553ede5` | Official H4; critique and Amendment 13 |
| `cface4d`, `32767e6`, `b8bdde8` | Official H2 (orig); official H1 + H2 A-arms + D-016; critique |
| `bcfe771`, `dc7cd54` | E5; critique |
| `5548384`, `90f0856` | E6; critique and Amendment 16 |
| `c475fbe`, `7172f38`, `9de73cd` | Open Images subset; E7; critique (HEAD when this report was written) |

---

## 8. Limitations and threats to validity

**Sample size and power**
- Per-generator test sets are small (54–62 test fakes per trained generator after E6's caption-family assignment, 131 FLUX; `015/results/tables.md`). Per-generator CIs are wide.
- The real test sets are small: 25 test phones, 183–199 RAISE test reals, 200 Open Images test reals. Open Images FPR has a resolution of 0.5%.
- No group is large enough to certify ≤ 1% FPR (§4.6).

**Real-photo coverage**
- VISION phones are pre-neural-ISP. The date range is given as 2015–2017 in `experiments/003-baselines-eval/notes.md` and as 2013–2016 in `experiments/007-worst-env-conformal/official/critique.md`; both are old.
- RAISE has no camera ids, so cameras cannot be held out, and its reals are bootstrapped per image (narrower CIs than a camera-clustered bootstrap) (`006/official/critique.md` F6).
- The Open Images files are re-encoded and resized copies from the CVDF bucket, not originals (`016/results/critique.md` F2).
- The "polished style" explanation of Open Images false positives comes from looking at about 30 images (`016/results/critique.md` F5).

**Fake coverage**
- All fakes share RAISE-caption content (LAION content only in E7's pilot-only arm). Content may be part of what any probe learned.
- Eight generators only. No video-frame, inpainting or image-editing fakes.

**Laundering coverage**
- Most laundering is simulated (`jpeg75_resize05`, `social_sim`, `screenshot_sim`, `mavali_medium`). The only real platform copies are VISION's WhatsApp and Facebook versions, which are content-confounded for DINOv2 (`014/results/critique.md` F2).

**Model coverage**
- One backbone (DINOv2-base). Larger backbones (DINOv2-L, DINOv3) were not tested, because of the download cap and DINOv3's gated licence (D-006, D-014).

**Adaptive reuse of the same test data**
- Many experiments reused the same official splits. Block 8 was chosen in E5 on the same images later used in E6. The E6 critic judged this minor, because the 4/8/12 concatenation gives the same result (`015/results/critique.md` F2), but repeated looks at one test set still inflate the best result.
- Several critic analyses are post hoc (for example, the Lumia 640 finding, the twin-status table, the Open Images montage). They explain results; they were not pre-registered tests.

**Process deviations (all disclosed in their files)**
- Amendment 4's H2-v2 rule was adopted after the pilot labels were seen.
- H2's α = 0.5% cell is a built-in tie that always votes NOT_SUPPORTED (`007/official/critique.md` §1).
- E1b and E3 ran on uncommitted code; bit-exact regeneration closed the gap (`docs/05` §3.6).
- H1's killed first launch was missing from the log until the critic found it (`005/official/critique.md` §1).
- Several agents were cut off by API limits mid-run; the lead read their outputs directly (`LOG.md`).

**Literature numbers**
- Literature figures quoted in this project come from `kb/` entries; some entries still carry [unverified] items (`kb/index.md`, `kb/verification-log.md`). This report uses no literature number as evidence for a result.

**The stop rule**
- The stop rule was overridden by the owner (D-016), and its limit was raised once for E7. Everything after the official H1/H2 results is exploratory and must not be read as confirmatory.

## Addendum (2026-10-26): E8 and E9, after this report was first written
- **E8, embedding geometry** (`experiments/017-embedding-geometry/results/`, critique.md). The real sources differ from each other almost as much as from the fakes (C2ST 0.977 vs 0.984). The fake directions per real source are only partly shared, so there is no single source-invariant linear "diffusion direction" findable with 2–3 sources in DINOv2 block 8. Projecting out the source subspace did not help (R3 KILL).
- **E9, real-anchor adaptation** (`experiments/018-real-anchor/results/`, critique.md). KILL.
  - The pre-registered A-off passed literally (0.957 / 0.985 / 0.951), but that was a label leak introduced by the lead's pre-registration: the shift was applied only to the held-out source's real test images.
  - The deployable reading (shift every image) leaves AUROC unchanged (0.917 / 0.689 / 0.809).
  - The label-agnostic A-proj gains +0.188 (RAISE) and +0.084 (OI) but stays below the bars: RAISE 0.877 [0.852, 0.899], OI FPR 8.3% [5.3, 11.7] at n = 50.
  - No label-agnostic cell meets the D-017 DISCOVERY tier.
  - The E8 critic's post-hoc few-shot lift, recomputed like for like (LOGO, n = 50), is RAISE about 0.879 and OI about 0.886. The earlier "0.936 / 0.933" were an oracle using all target reals with full-model scoring.
- **Conclusion.** The real-anchor line is closed (critic recommendation; no further limit raise). The central obstacle stated in §4 stands: detectors learn the real-photo pipeline, and a few anchor photos only partly correct it (≈ 0.88 AUROC, 4–8% FPR on unseen sources).
