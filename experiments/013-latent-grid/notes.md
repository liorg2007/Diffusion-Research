# 013-latent-grid: H-D2 PILOT (latent-grid periodicity under rescaling)

Spec: `kb/ideas/latent-grid-periodicity-rescale.md` (ca58d61) + `docs/04-phase4-plan.md` Amendment 12. Operationalisations were fixed in `config.json` before the scan. Diagnostic under D-014, no stop-rule weight. Nothing is trained.

Run: `selftest.py` -> `run.py --stage scan` (CPU, 1m46s) -> `--stage e4` (GPU, about 12 min, 8.5 min of it the first DINOv2 pass) -> `--stage analyze` (9 s) ; `positive_control.py` (post hoc). Outputs in `pilot/` (results.json, tables.md, scores_long.csv.gz, d008_guard.json, e4_addon.json, positive_control.json, run.log).

## Design as run
- Data: orig arm (plain unresampled 512 centre crops; SD2/SDXL crop offsets are multiples of 8, so the latent grid has phase 0). 358 test natives (25 devices) + 358 WA + 358 FBH test copies; 720 fakes (120 x 6 generators). fov512 resamples fakes (s_fov = 0.50-0.89 for SD fakes), so it is descriptive.
- O (oracle s) and S (blind) as in `config.json`; folded-grid energy z8 (p = 8) and zs (p = 8s when integer); Laplacian texture baseline (better direction); JM control (SD fakes JPEG-saved at a paired native's QF 88-100 before the condition); conditions: clean, JPEG-only, resize-only 0.5/0.6/0.7/0.75/0.8, jr(75,s) for the same s, social_sim.
- E4 add-on: 011 pair setup refit here (identical to 011 at jpeg75_resize05, max diff 0), DINOv2 + F3 recomputed at jr(75,0.7); D-008 max |dlogit| 2.4e-4 (<= 1e-3).

## Results (source: pilot/results.json, pilot/tables.md)
- **Validity gate FAILS -> label UNINFORMATIVE.** Primary reading: share of SD2+SDXL clean fakes with the O prominence at 1/8 above the 99th percentile of clean natives (1.51 log10) = 0.000 (SD2 0.000, SDXL 0.000; natives 0.011). Secondary reading (argmax at 1/8): 0.013 (natives 0.059). Median O at 1/8: SD2 0.385, SDXL 0.376, natives 0.433. There is no coherent 8-px peak to track in unresampled SD2/SDXL fakes under this score.
- O-AUROC (SD2+SDXL vs natives, jr(75,s)): s=0.5 0.528 [0.485, 0.573]; 0.6 0.488 [0.439, 0.539]; 0.7 0.504 [0.458, 0.554]; 0.75 0.487 [0.441, 0.534]; 0.8 0.501 [0.454, 0.551]; JPEG-only s=1 0.514 [0.464, 0.562]; clean 0.462 [0.411, 0.514]. All at chance (the fail rule "O < 0.70 at all non-dyadic s" would also trigger, but the gate takes precedence by the pre-declared label order).
- Dyadic gap mean O(0.6,0.7,0.8) - mean O(jr050, jpeg75) = -0.023 [-0.052, +0.005] (fail rule < 0.05 also triggers): no collision effect is visible because there is no signal to collide.
- JM control: |O_JM - O| <= 0.01 at every condition (no format confound visible; uninformative while O is at chance).
- Texture baseline (Laplacian, better direction): 0.72-0.83 for SD-family vs natives at all conditions; O is 0.20-0.37 below it.
- S at social_sim: SD-family 0.483, all 6 0.465 (needs 0.80 / 0.70). Folded-grid z8 (SD): 0.46 clean, 0.49 JPEG-only, 0.51-0.62 for resized conditions (0.60 [0.54, 0.66] at jr050; zs at 0.5x 0.634 [0.575, 0.696], at 0.75x 0.643 [0.568, 0.719]); weak, may mostly reflect resize/texture differences, not the grid.
- FPR of S (99th percentile of clean natives = 1.329): WhatsApp 0.008, FB-HQ 0.011 (clean copies) -> within 2%, trivially, since S is at chance.
- fov512 (descriptive): O (SD) 0.55-0.63, S ~0.5, z8 0.56-0.68, texture 0.61-0.64. The fov-resampled SD fakes put the grid within 1 bin of k/8 for 3% (clean), 19% (extra 0.5x), 0% (0.7x), 15% (0.75x) of images.
- **E4 add-on** (orig, pair AUROC, primary / matched, jpeg75_resize05 -> jpeg75_resize07): F2 0.597 -> 0.672 / 0.670 -> 0.703; F4 0.598 -> 0.671 / 0.670 -> 0.699; F1 0.560 -> 0.591; F3 0.527 -> 0.513 / 0.569 -> 0.567. Best primary 0.672 (F2), best matched 0.703 (F4/F2). Win-rates 0.84 -> 0.93 (F2 primary). So the 0.7x condition recovers about 0.07 AUROC over 0.5x, but the DINOv2-based families, not the periodicity family F3 (which does not improve), carry it; the descriptive threshold (>= 0.70 primary) is not reached by the primary probe, only by the matched one (0.703, CI [0.671, 0.741]). fov512: F2 0.599 -> 0.674 primary, 0.739 -> 0.750 matched.
- Post-hoc positive control (`positive_control.py`, not part of the criteria): known sd-vae-ft-mse round-trips of 150 fov512 test reals vs the reals: O AUROC 0.522, and 011's own F3 exact-bin peak AUROCs are 0.52-0.55 at 1/8 (x/y), 0.60-0.62 at 2/8, 0.62-0.63 at 3/8, 0.64-0.74 at 4/8. So even the known VAE trace is not a coherent 1/8 line in this residual; it is stronger at the higher harmonics (period 2-4 px) and in other features. The negative gate is therefore not a bug of the O score (it matches 011's independent F3 numbers), and the selftest shows the score finds an injected 8-px grid and moves with 1/(8s).

## Interpretation (labelled)
UNINFORMATIVE by the pre-declared gate (stated reading). The hypothesis H-D2 is not supported on this data: no fundamental-period grid is measurable in SD2/SDXL fakes, with or without resize. This does not test the dyadic-collision prediction. A follow-up on the higher harmonics (periods 4 and 2 px, i.e. 2/8-4/8) would be a new hypothesis, not part of this pilot.

## Self-test (selftest.py, 6 pass)
Peak at 1/8 for an injected 8-px grid; peak at 1/(8 s) for s = 0.8, 0.75, 0.7, 0.6, 0.5; blind S ignores k/8 and finds a 5.6 px period; fold energy; JPEG lattice at multiples of 1/8 of the output grid for every s; conditions' sizes. Deviation: on synthetic content NO dyadic-specific AUROC loss appears after JPEG q75 (median-residual removes block steps; grid survives at every s, AUROC 1.0; at q20 it is lost at all s), so the collision itself is not reproducible synthetically; only the dyadic geometry is asserted.

## Deviations / ambiguities
- Gate "peak at 1/8 +-1 bin" operationalised as above (pre-declared in config.json; both readings reported, both fail).
- O uses the average of x and y axes, Hann window, max over +-1 bin; S uses the combined curve. FPR rule read as clean natives vs clean WA/FBH copies (threshold from the same 358 test natives; the copies are of those photos).
- JM JPEG lattice is phase-aligned with the crop origin; natives' capture lattice has random phase.
- Bootstrap: 2000 replicates, reals by device, fakes by image per generator.

## Lead correction after the critic (2026-10-24)
The statement "there is no 8-px grid in SD fakes" is WRONG (`pilot/critique.md`). An 8-px lattice is present in about 70% of SD2/SDXL fakes (fold z8 about 2.7 vs about 1.1 at periods 7/9). Only about 10% of its energy is at the 1/8 fundamental the O score measures; the rest is at period 2–4 px harmonics and off-axis. VISION natives carry a capture-JPEG grid on the same lattice, so grid energy does not separate them from SD fakes (AUROC 0.44–0.47).
The pre-registered label UNINFORMATIVE stands. The dyadic-collision mechanism is "not supported (post hoc)": the trace decays smoothly with downscaling and has no dip at 0.5×.
