# H-D1 PILOT critique (self-referenced decoder-trace contrast), independent critic (opus/high), 2026-10-25

**Rating: SOUND, no bug found. The computed label (UNINFORMATIVE; NOT DIFFUSION-SPECIFIC) follows the pre-registered rules correctly. Two readings need correcting:**
- **"UNINFORMATIVE" understates the result.** Ignore the gates and score B on the pass/fail rules: it is a clear FAIL in both arms. B laundered is 0.496 in fov512 and 0.484 in orig, against a fail line of 0.70 and against F2p + 0.03.
- **B is at chance at clean fov512 because of the arm, not the code.** The fov512 preparation resamples every fake, so no fake is a decoder fixed point any more. Real and fake shifts are then identical: median 0.549 vs 0.551. In orig (native pixel grid) the mechanism does appear at clean: B is 0.704, and SD-family 0.867. It disappears under laundering in both arms. That is the real refutation.

PILOT under D-014: none of this counts toward the stop rule. Every check below is post hoc and descriptive. All ran on CPU and read existing caches only (no download, no VAE pass). Scripts, logs and JSON are in `pilot/critic_checks/`:
- `c1_scores.py`: shift distributions per class and generator; AUROC recomputation from `scores_long.csv.gz`;
- `c2_feats.py`: feature identity with E4's training features; shift vs content spread;
- `c3_blur_probe.py`: a blur-trained probe as a control for the side number. DINOv2 ran on CPU; its features match the CUDA cache (max |Δφ| 6.2e-05).

## Question 1: bugs, and why B is at chance even at clean

### No bug found
- **B is computed as pre-registered.** `compute_scores` uses B = −(d_m(φ(L0 VAE t′)) − d_m(φ(L0 t′))):
  - d_m is E4's matched probe (fit on `jpeg75_resize05` pairs);
  - L0 is applied to both the observed t′ and VAE(t′), after the VAE;
  - the orientation is fixed on the train pairs.
- **The refit probes reproduce 011 exactly** (abs diff 0, `check_probes_vs_011.json`).
- **B at clean fov512 is exactly E4's matched-pair quantity.**
  - For the 297 train reals, 012's φ(L0 t) equals 004's laundered cache (max |Δ| 3.3e-05).
  - 012's φ(L0 VAE t) equals 011's laundered round-trip features (max |Δ| 3.4e-05) (`c2_feats.json`).
  - So the clean failure is not a pipeline mismatch.
  - In orig the round-trip features differ (max 3.2, median 0.004). This is disclosed deviation 2: there is no re-save as JPEG at the source QF. It affects only the secondary arm.
- **L0 at clean is not the identity.** L0 always runs `jpeg75_resize05`. In fov512 it takes 512 to 256, with no upsampling under the `min` rule. The median relative change in φ for natives is 0.59 (`c2_feats.json`).
  - Bu upsamples back to 512 and is also at chance (0.557 clean). The size rule is therefore not the cause.
  - At laundered, Bu equals B exactly (Δ = 0.000), as expected: both rules upsample a 128-px image.
- **Gates are evaluated as written.**
  - Gate 1 passes: ratio −0.029 in fov512, −0.141 in orig.
  - Gate 2 (Spearman of d_m(φ(L0 t)) vs d_m(φ(L0 VAE t)) on test natives) fails in fov512 at clean (0.671) and at laundered (0.785). It passes at social (0.862).
    - It still fails under any single-condition reading that uses the label condition.
    - It passes in orig in all three conditions.
  - Gate 5 (B clean > texture clean) fails in both arms: 0.533 vs 0.606, and 0.704 vs 0.772.
- **Blur σ matching works as specified.**
  - All 12 (arm, condition, score) curves cross the VAE target exactly once.
  - The A-curves are non-monotone in four cells, but the first crossing is unique.
  - Matching uses the mean shift of the train reals, as pre-registered.
- **The bootstrap is correct.**
  - One paired plan for all scores: devices for reals, images within each generator for fakes.
  - The per-pool plans share draws (asserted).
- **The macro AUROCs recompute exactly** from `scores_long` (e.g. B 0.533 / 0.496 / 0.539, `c1_scores.log`).
- **The label logic is correct.**
  - A failed gate gives UNINFORMATIVE.
  - The blur rule then compares B with blurB at laundered: −0.131 [−0.170, −0.093]. The blur is not beaten.
  - The D-008 probe guard passes: max |Δscore| 1.1e-04.

### F1 (HIGH): mechanism of the clean-fov512 null — resampled fakes are not fixed points
- **Shifts along the matched direction** (median d_m change on a round-trip, `c1_scores.log`):

  | | reals (natives) | fakes |
  |---|---|---|
  | fov512 clean | 0.549 | 0.39 to 0.90 per generator; 0.551 pooled |
  | orig clean | 1.51 | SD2 0.18, MJ5 −0.03, SDXL 0.74; DALL-E 2 / DALL-E 3 / Firefly 1.70 / 1.49 / 1.37 |

  - In fov512, 94–100% of fakes shift positively, like reals.
  - In orig, the SD2 / MJ5 / SDXL fakes are near-idempotent; DALL-E 2, DALL-E 3 and Firefly move like reals.
- **The cause.** fov512 bicubic-downsizes every Synthbuster fake (short side above 512, down to 512; exact source sizes per generator [unverified] here). That moves it off the decoder's pixel grid, so VAE(t) re-renders it just as it does a real photo.
  - E4's 0.98 within-pair win-rate only says that reals move. It never said that fakes do not, so it was never evidence for B.
  - The E4 critic had already flagged this ("loses at fov512, where the fakes have been resampled", `011/pilot/critique.md` F3).
- **Signal vs content.** In fov512 the matched-direction shift has Cohen's d = 0.36 between the classes. Its spread within reals (SD 0.86) is about as large as the single-image spread of d_m(φ(L0 t)) (SD 1.19).
- **SD2 is inverted in fov512** (B 0.390 clean, 0.311 laundered). Its fakes move more than reals (median 0.90). A plausible reason is that SD2's fakes are downsized the least, so they keep more re-renderable detail than the reals [unverified].
- **Why A still gets 0.795 at clean fov512.** A is the primary-direction shift, with no L0. Fakes move less than reals (6.5 vs 9.5), but blurA scores higher (0.838; A − blurA = −0.042 [−0.084, −0.001]). In this arm A therefore measures how much detail a low-pass removes, not idempotence.
  - A is inverted at laundered (0.434): laundered fakes move more (3.88 vs 3.20).
- **This is a pre-registration design flaw, not a code fault.** The label arm (fov512, chosen for zoom hygiene) structurally removes the property the hypothesis needs. No gate checked that fakes in the label arm are still on the decoder grid: gate 1 tests idempotence on reals' own round-trips only.

### F2 (HIGH): the laundered failure is general, so the hypothesis is refuted, not merely uninformative
- At laundered, B is 0.496 (fov512) and 0.484 (orig). In orig, fakes' median shift is now *above* that of reals (1.37 vs 1.26), and the class-level d is 0.04 (`c2_feats.json`).
- After ×0.5 and JPEG, no fake is a decoder output in either arm. Applying L0 to both sides cannot bring back a fixed point that resampling destroyed.
- **The idea file's central premise is falsified.** It claimed that a decoded image is a fixed point "in the mid-frequency band that survives laundering". This matches the E4 critic's preview: native-scale clean only.
- **Every pass criterion misses by a wide margin.** B laundered ≥ 0.80 is missed by 0.30. B ≤ F2p + 0.03 (0.665): B is 0.17 below F2p. Non-SD laundered is 0.531.
- The only thing that turns this into UNINFORMATIVE is gate 5 ("B beats texture at clean"). That is a quality bar, and failing it is itself evidence against the idea.

## Question 2: is "not diffusion-specific" correctly concluded?
- **Formally, yes.** The pre-registered contrast is B − blurB at laundered: −0.131 [−0.170, −0.093] in fov512 and −0.047 [−0.112, +0.014] in orig.
- **Substantively, the label is near-trivial.** B carries no signal at laundered, so it cannot beat anything.
  - blurB's laundered 0.628 is a texture-like score: texture gets 0.621 in the same cell.
  - Read the label as "no laundered signal", not as "a signal that turns out to be generic".
- **Do not generalise it to "the VAE reference has no decoder-specific content".** At clean orig, B beats its matched blur by +0.368 [+0.306, +0.434]:
  - per generator, B / blurB: SD2 0.930 / 0.272, MJ5 0.954 / 0.287, SDXL 0.804 / 0.456;
  - DALL-E 2, DALL-E 3 and Firefly are at chance for B.

  This is a decoder-family flag at native scale on clean images, as the E4 critic found. It sits outside the label condition and outside the label arm.

## Question 3: the side number (pooled F2: 0.892 / 0.665 / 0.759 in fov512)
- **Sound as a number.** It is E4's primary probe, trained on real vs own SD-VAE round-trip pairs (297 VISION reals, no generated images), scored on φ(t). It reproduces 011's results exactly. The E4 critic's 60-per-generator sample gave 0.887 (F4).
- **Not a low-pass axis (new, `c3_blur_probe.json`).** The same recipe trained on reals vs Gaussian-blurred reals transfers *inverted*:

  | blur σ | clean | laundered | social_sim |
  |---|---|---|---|
  | 0.867 | 0.401 | 0.434 | 0.431 |
  | 2.476 | 0.388 | 0.431 | 0.423 |

  - The weights are nearly orthogonal to the VAE probe (cosine 0.01 / 0.11), and the scores are uncorrelated with F2p (ρ ≈ −0.05).
  - So F2p's clean 0.892 is not "smoother = fake". The SD-VAE round-trip teaches something blur does not.
  - It also transfers to non-SD generators (DALL-E 2 0.922, DALL-E 3 0.900 clean), so it is not shown to be decoder-specific either.
- **It is no better than E2 #6 under laundering, and far below B0.**

  | method | arm | clean | laundered |
  |---|---|---|---|
  | F2p | orig | 0.776 | 0.643 |
  | E2 #6 A (`docs/05`) | orig | 0.819 | 0.778 |
  | F2p | fov512 | 0.892 | 0.665 |
  | E2 #6 A | fov512 | — | 0.681 |

  - In fov512 the two tie at laundered. These are separate runs, so there is no paired CI.
  - B0 (trained on generator images, not generator-free): LOGO 0.992 clean / 0.985 laundered (005 pilot, in `results.json`), and about 0.98 laundered in fov512 (`docs/05`).
  - At the 1% threshold, F2p's TPR is 0.324 clean but 0.033 laundered (device cross-fit).
- **Confounds to state whenever it is quoted:**
  - **Zoom and scale.** The probe was trained on 224 crops of 512-px images. Laundered test images are 256 px, so the crop covers about twice the field of view, and the 0.892 → 0.665 drop mixes laundering with a change of scale. The matched probe is worse on single images (0.626), so scale mismatch is not the whole story.
  - **Downscale ratio.** In fov512, reals are shrunk about 4–8× and fakes about 1.5–4× [unverified per image]. This is a residual resampling cue present in every fov512 number.
  - **Camera prior.** The training contrast is "camera photo vs processed copy". #6 rode a similar axis and failed on RAID web reals (0.52, `docs/05`). F2p has not been tested on non-camera reals.
  - **Selection.** fov512 F2 is the best of E4's families and arms. It was pre-registered as a baseline here but not as a method.
- **Verdict.** Credible as the standard *generator-free reference baseline* at clean. Not a candidate method, because the whole gap is laundering.

## Question 4: implications for D-014 (pre-registerable recommendations only)
1. **Record the result.** Write H-D1 as "UNINFORMATIVE (pre-registered label); would be FAIL on every criterion; mechanism falsified under resampling". Retire the post-laundering self-reference score. Do not retune L0 rules, w or σ on these data.
2. **Reconstruction-type evidence.** Three reconstruction or idempotence scores have now failed under laundering:
   - #7 VAE ratio (0.318);
   - the E4 critic's Δs (0.45–0.56);
   - H-D1 B (≈0.49).

   Pre-register that D-014 stops spending pilots on fixed-point scores for laundered inputs. The one remaining use is a clean, native-scale **decoder-family flag**: orig B SD-family 0.867. If it is pursued, it is a component with its own gate: "fakes have not been resampled since decoding".
3. **Mechanism gate for any future arm choice.** The label arm must preserve the property the hypothesis needs. Pre-register a check on fakes before reading any AUROC; for example, median fake shift ≤ 0.5 × median real shift in the label arm at clean.
4. **Separate mechanism gates from quality gates.** A quality bar such as gate 5 should not turn a clear FAIL into UNINFORMATIVE. Proposed rule: if the score misses the fail line by ≥ 0.10 whatever the gates say, the label is FAIL.
5. **Adopt pooled F2 (E4 primary) as the generator-free baseline** in D-014 experiments, beside texture and B0. Pre-register three controls for it:
   - the blur-trained probe (c3: 0.39–0.40);
   - a cross-scale check: the fov512 probe on orig-arm crops;
   - an FPR check on non-camera reals (RAID / web reals) before any claim that it detects decoders.
6. **Where effort can move.** The remaining gap is laundering robustness of a single-image trace: F2p 0.665, matched 0.626, #6 0.681, against B0 at about 0.98. Pre-register a laundering- or multi-scale-augmented VAE-pair probe against F2p, with the rule "≥ +0.10 laundered fov512, CI lo > 0".
7. **Procedure.**
   - `run.py`, `config.json` and the results went into one commit (a81cfed). Commit code and config before the full run, so that "fixed before the run" can be checked in git. No outcome here depends on it: every variant fails.
   - `notes.md` has no results section. The lead should add a pointer to `tables.md` and to this critique.
