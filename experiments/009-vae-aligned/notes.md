# 009-vae-aligned: E2 PILOT (docs/04-phase4-plan.md Amendment 8)

STATUS: code written and compiled; NO result yet. The owner asked for no GPU-heavy work, and the lead terminated the first
`--stage timing` run. Nothing has been computed: no timing gate, no round-trips, no features, no D-008 guard.

## Resume (when the lead allows GPU use)
    env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage timing    # 20-image gate -> pilot/timing.json (cut decided here)
    env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage guard     # D-008 VAE CUDA vs CPU (32 images; CPU is slow); vae and analyze refuse to run unless it passed
    env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage vae       # 594 round-trips per arm + #7 e1/e2 (resumable)
    env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage feat      # DINOv2 features of round-trips (004 pipeline)
    env/.venv/bin/python -I experiments/009-vae-aligned/run.py --stage analyze   # CPU probes, metrics, labels, tables + probe D-008 (CPU DINOv2 features)
(`--stage all` runs them in that order.) The analyze stage needs no GPU except that D-008 compares against the cached GPU features.

## Known issue seen in the one aborted run
`accelerate` is not installed, so diffusers loads the VAE with `low_cpu_mem_usage=False` (a warning, slow load). Not installing
anything (CLAUDE.md); the run was killed before the load finished, so it is not known whether it merely loads slowly.

## Interpretation choices (all stated in config.json)
- VAE: latent_dist.mode() (posterior mean), fp32, TF32 off, uint8 quantization of every round-trip; batch <= 4 with OOM halving.
- A trains on CLEAN features only (A0 convention); conditions are applied at test time, identically for both classes.
- #6 evaluation set follows 005 LOGO: 358 test natives vs all 120 fakes per generator (features are cached, no VAE cost).
  #7 uses the split == test rows (358 natives, 60 fakes x 6), cut to 150 / 30 per generator only if the timing gate fails.
- #7 laundered condition: the VAE runs on the conditioned image (256x256 for jpeg75_resize05); ratio in [0,1] float space, whole image L2.
- Texture-baseline sign: calib natives vs train-split fakes (the pilot has no calib fakes; 008 used the same workaround).
- C = B's LOGO rows + 594 aligned fakes (as written). C_all_reals additionally adds the 198 WA/FBH train reals (sensitivity,
  because otherwise the WA/FBH-style aligned fakes have no matching reals).
- "Beats the texture baseline" is read as: ratio macro AUROC > texture macro AUROC at both clean and jpeg75_resize05 (point estimates).
- D-008 VAE guard measures the float output before quantization; x2 is computed from the same quantized x1 on both devices.

## Tests
`CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/lib/tests/run_all.py`: 58 passed, 0 failed (CPU only).

## Fixes applied after the pre-run code review (code_review.md; docs/04-phase4-plan.md "Amendment 7-8 clarifications")
1. #7 validity gate: valid iff the CI lower bound of the clean ratio AUROC is > 0.5 (below chance = UNINFORMATIVE); both readings stored in results.json.
2. fov512 arm: aligned fakes saved as PNG (orig: JPEG at the source QF). All cache writes are atomic (tmp + os.replace).
3. CV groups = source photo `device:vidx` (fakes: sha256); a round-trip carries its source real's group. B uses the same groups (005 used sha256; the B-vs-005 reproduction is reported, not asserted).
4. Order timing -> guard -> vae -> feat -> analyze; `vae` and `analyze` refuse to run unless the VAE guard passed; the probe guard runs before results.json (nothing is written if it fails).
Self-test (CPU, no GPU): `CUDA_VISIBLE_DEVICES="" env/.venv/bin/python -I experiments/009-vae-aligned/selftest.py`.
