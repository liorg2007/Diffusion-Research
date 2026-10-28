# H-A6 critique (opus/high critic, 2026-10-28). Non-counting diagnostic

**Verdict: label INCONCLUSIVE is correct under the rule frozen in config.json. In practice it is a KILL for any laundered use and a clean-only PASS.** Clean results are real and come from decoder matching, not from native-size cues. One D-008 finding needs the lead's attention (point 4). Scratch checks are in experiments/024-attr-vae/critic/ (scripts plus .txt outputs, CPU only, cached features, and one 16-image sd-vae CPU run).

## 1. Label
- config.json `label_rule`: PASS needs jpeg75_resize05 ≥ 0.50 with clean-train; that arm is 0.333, so no PASS. KILL needs < 0.45 under BOTH sources; matched is 0.584, so no KILL. Result: INCONCLUSIVE.
- docs/08 states KILL as "< 0.45 at jpeg75_resize05" without naming a source. The "both sources" reading was frozen before any result, so it is legitimate. Under the realistic clean-train reading, the result would be KILL.
- Matched-train (0.584 / 0.591) is below M1 texture (0.735 / 0.701; diff CI fully < 0). Gate 2 therefore gives the VAE features no laundered value even with matched training. Recommended wording: "INCONCLUSIVE by rule; laundered = no signal beyond texture".
- Laundered collapse is a lost trace, not a scale shift that recalibration could fix. A2 is the training-free within-image VAE comparison, so it cancels scale. It drops from 1.000 to 0.453 / 0.484 (tables.md). Under jpeg75_resize05 every generator prefers sdxl-vae, SD2 included (sd-vae preferred by only 0.12 of SD2; critic/check_size.txt).

## 2. Is clean driven by native size or crop margin? No, for SD2 vs SDXL
- The notes say "SD2/SDXL in Synthbuster are 576-768 px". That is **wrong**. Both use the SAME 26 aspect buckets (short side 576–1024, long side 704–1728, 0.95–1.05 MP; checked from data/manifest_raw.csv). So size, crop fraction and crop margin are matched between the two classes.
- The SD2 / SDXL preference is 100% / 99% in **every** size bucket (critic/check_size.txt). Mean log AEDR: SD2 0.31 (sd-vae) vs 0.79 (sdxl); SDXL 0.85 vs 0.42. This is a genuine decoder-match signal.
- The 6 "other" generators are all ≥ 1024 px and their crops have content zoom similar to the SD2/SDXL crops. Most sit off both manifolds: log AEDR_sd 0.75–0.88 for dalle2/3, firefly, flux and sd3. **The exception is MJ v5:** all 16/16 images prefer sd-vae, at log AEDR 0.26, lower than SD2. The LR's 3 "other" → kl-f8 errors are 3 of the 4 MJ test images. Without MJ, clean bacc is 0.991 (critic/check_pergen.txt). So the class "kl-f8" really means "decoder close to sd-vae-ft-mse", and MJ v5 seems to use such a decoder (inference, [unverified]). This also bears on the H-A1 MJ↔SD confusions.

## 3. SD1.x → "other": not a crop-margin effect; likely decoder weights
- SD1.3 / 1.4 (512 px, full frame) sit off BOTH manifolds: log AEDR_sd 0.79 vs AEDR_xl 0.80, and the A2 preference is a coin flip (0.46 / 0.44).
- **Border control.** An interior 448 crop at (32,32) has no image border and keeps the 8-px phase. On 8 SD1.4 images it gives log AEDR_sd 0.67–0.93 (mean 0.80); on 8 SD2 images, 0.20–0.47 (critic/check_border.txt). These are unchanged from the full-crop values, so the missing crop margin is ruled out.
- A residual caveat remains: content zoom. A full 512 scene is denser than a crop of a 1 MP image. It is unlikely to explain the gap, because AEDR is flat across SD2 zoom levels and across the 1024 px "other" generators.
- More parsimonious explanation: sd-vae-ft-mse is a decoder fine-tune, while SD1.1–1.4 shipped the original kl-f8 decoder; whether SD2 ships an ft-family decoder is [unverified] (docs/07 part 2 l.29). Under that reading, the signature identifies decoder *weights*: one decoder fine-tune is enough to break the "SD kl-f8 family" mapping.
- Consequences: the "SD2 → kl-f8" label is empirically justified for SD2 (120/120 on-manifold), but the family name overstates it. The notes' SD1.x sentence should read "SD1.x does not match sd-vae-ft-mse (decoder-specific), border ruled out; [unverified] which decoder SD1.x/SD2 ship". The original CompVis kl-f8 decoder would settle it; it is not downloaded.

## 4. D-008 handling: correct by the letter, but the guard is ill-posed for sdxl
- The capped analysis uses all-CPU features (the reference device), so formally D-008 does not apply. Subset selection is consistent: cap_subset and the SD1.x choice are seeded identically in extract and analyze, and feats_matrix raises KeyError on any missing key.
- **New finding.** The 18 guard images that are also in the capped set were extracted twice on CPU, by the guard run and by the main run. sdxl features differ **CPU vs CPU** by up to 1.1e-2, the same size as the CUDA-vs-CPU gap (1.15e-2) that "failed" D-008. The primary LR's logits move 0.03–0.12 between the two CPU runs, with 0 prediction flips (critic/check_cpu_repro.txt). The likely cause [hypothesis] is uint8 re-quantisation between passes amplifying float noise in log e2 / AEDR, plus batch or thread-dependent numerics.
- So the 1e-3 logit guard measures this feature's own instability, not GPU error. The cap from 1,020 to 360 images (with 7 h of CPU) was probably unnecessary.
- Recommendation for the lead (no rule change by me, since D-008 is a CLAUDE.md rule): log this. For future quantised-round-trip features, compare accelerator-vs-CPU against CPU-vs-CPU repeat noise, and report decision flips.
- The sd-vae-only 1,020-row table uses GPU features. Its logit guard was 1.5e-3–4.3e-3, above 1e-3, and was measured on the 8-feature model, not on that table's model. Mark it "D-008 not met, descriptive" rather than "marginal".

## 5. Sample size
- Test set: 108 images (34 / 33 / 41). The 41 "other" images are 4–10 per generator, MJ only 4. The clean CI [0.933, 0.993] and split-seed 0.950 ± 0.024 support the 3-way clean claim. Per-generator "other" statements are anecdotal.
- A2 = 1.000 on 67 test images, plus a 120/120 (SD2) and 119/120 (SDXL) preference over the whole capped set, is robust.
- The SD1.x arms (50 each) are adequate for "0% kl-f8 on clean". The matched-laundered shares (CI widths about 0.25) are too wide to claim "weakly in favour". Drop that phrase.

## 6. Requested edits to notes.md (lead merges)
1. Correct the SD2/SDXL size sentence (identical 1 MP buckets).
2. Add the MJ v5 on-manifold observation.
3. Add the border control result.
4. Re-tag the sd-vae-only 1,020 table as D-008 not met.
5. Add the CPU-vs-CPU nondeterminism to the D-008 section.
6. Tick the critique gate.

## 7. What H-A6 adds to the cycle
H-A6 shows that training-free VAE attribution works, but only on pristine files.
- **What works.** On native-grid PNG crops, comparing round-trip AEDR across two candidate decoders identifies the decoder exactly: SD2 vs SDXL 1.000 with no training. Size is matched, so this is a real decoder trace, and it beats M1 by +0.21 (CI lower bound +0.13).
- **What it is specific to.** It recognises a decoder's weights, not its architecture family. SD1.x falls off the sd-vae-ft-mse manifold, while MJ v5 falls on it. So a bank of decoders answers "which public VAE decoded this", not "which product made this".
- **What breaks it.** One jpeg75 + 0.5× resize (or social_sim) erases the trace: A2 drops to chance and the clean-trained model outputs "other" for every image. Matched training stays below texture.
- **What this means for the project.** VAE reconstruction is a forensic tool for untouched files and provenance chains. It is not a candidate for the laundered, in-the-wild goal. This is consistent with the earlier laundered failures (03-final-report §4.2) and the docs/08 prediction.
