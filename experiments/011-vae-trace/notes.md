# 011-vae-trace: E4 PILOT (docs/04-phase4-plan.md Amendment 11, D-014)

Diagnostic: can a probe separate a real photo from its OWN sd-vae-ft-mse round-trip, which feature family sees the trace, and does it
survive `jpeg75_resize05`? Does not count toward the stop rule. Results: `pilot/results.json`, `pilot/tables.md` (see there; numbers are
not repeated here so that this file cannot drift from them).

## Run
    env/.venv/bin/python -I experiments/011-vae-trace/selftest.py            # CPU, synthetic F3 peak test etc.
    env/.venv/bin/python -I experiments/011-vae-trace/run.py --stage all --dry   # tiny subset -> dry/ (plumbing only)
    env/.venv/bin/python -I experiments/011-vae-trace/run.py --stage all     # rt -> feat -> analyze (resumable from cache/)

## Interpretation choices (all fixed before any result; also in config.json)
- Pair AUROC = pooled Mann-Whitney AUROC of probe scores, round-trip (high) vs real, over all test pairs (the quantity E2's critic reported
  as 0.66 / 0.56). The within-pair win-rate P(score_rt > score_real) is added descriptively.
- Probe training: PRIMARY = fit on CLEAN train pairs, applied to both test conditions (A0 / E2 convention). The decision rules are applied
  to it. MATCHED (fit on pairs of the tested condition) is a sensitivity row, not pre-registered; its rule outcomes are printed beside.
- Best family = argmax of the test point estimate over F1..F4 (4 candidates chosen on test: mildly optimistic; all four are listed).
- Trace-specific is evaluated at both conditions; the headline is clean.
- F3 is computed on the 224 center crop of the conditioned image (the DINOv2 input crop; 224 and the crop offsets are multiples of 8, so
  the f = k/8 bins are exact FFT bins). No window. "Periodicity peak" = log10 power of the exact bin minus log10 mean power of the
  6 neighbours (+-2..4 bins) on the same axis (fy=0 row for x, fx=0 column for y). A single FFT bin is chi-square noisy (white noise gives
  sd ~0.4-1 in log10; the Nyquist bin f=4/8 is the noisiest), so the self-test judges the injected 8-px pattern against that spread.
- Transfer: the pair-trained probe is scored on the 358 test natives vs ALL 120 Synthbuster fakes per generator (E2 convention; the probe
  never sees a generator image). SD-family = SD2, SDXL; other = DALL-E 2/3, Firefly, Midjourney v5. The Laplacian baseline's direction is the
  one fixed on train PAIRS (round-trip vs real), not re-tuned for transfer.
- Texture baseline sign: per condition, from train pairs, never flipped on test.

## Data handling
- Train round-trips (297 per arm) = 009's cache (read-only). Verified against the Amendment 7-8 rules: orig = JPEG at the source QF
  (`jpeg_quality_estimate` of each cached file equals the source QF), fov512 = PNG, all 512x512; 16 per arm regenerated here and compared with
  the cached files; and our DINOv2 features of them compared with 009's cached ones (`pilot/check_009_cache.json`, `check_009_feats.json`).
- Test round-trips (1,074 per arm) are generated here in `cache/rt/<arm>/<sha>.{jpg,png}`; QFs in `cache/qf.json`.
- Real features: cached 004 features (`p4features.load_features_aligned`, subdirs pilot / pilot_fov512). Round-trip DINOv2 features and
  F3 / Laplacian for everything are computed here (`cache/feats/`).
- Pair identity and clusters: CV group = `device:vidx` (the native, WA and FBH copies of a photo and all their round-trips share a fold);
  bootstrap cluster = device, and the real and its round-trip carry the same weight.

## Caveats stated in advance
- The orig arm's round-trips are fresh JPEGs on a crop-aligned 8x8 block grid, while a real crop's JPEG grid (if any) is generally not
  aligned to the crop. An F3 8-px peak in the orig arm can therefore be a JPEG-grid cue, not a VAE cue. The fov512 arm (PNG for both) does
  not have this cue and is the cleaner read of F3's periodicity features.
- 25 test devices, 10 train devices: the CI reflects device sampling, not new device types.
- VISION native/WA/FBH reals only; no RAISE-like PNG reals.

## D-008
- VAE: 009's pixel guard for sd-vae-ft-mse is reused (same revision and function); see `pilot/d008_guard.json` ("reused_from").
- Probes: 32 test round-trips per arm and condition, DINOv2 CPU vs CUDA, max |dlogit| over all probes <= 1e-3, checked before results.json is
  written. `OMP_NUM_THREADS=4` is set in run.py (16 BLAS threads on a loaded box made sklearn 25x slower; not a numerical change beyond fp noise).
