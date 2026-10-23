# Pre-run code review: H4 OFFICIAL (`run_official.py`, `config_official.json`, `lib/h4.py`, 004 `fov_prep.py` @ 82b5026)

**Verdict: READY WITH FIXES.** The label logic, contrasts, direction, pairing and data splits match H4 plus Amendments 3, 5 and 6. No sign errors were found, and there is no test leakage. Before the GPU run, fix #2 and #3 and decide #1. The other findings are disclosure or hardening.

Reviewer: independent pre-run code review, 2026-10-24. CPU only, no GPU, no full run.
Checks that ran:
- `experiments/lib/tests/run_all.py`: 58 passed, 0 failed (includes the 7 `test_h4.py` tests).
- Read the existing standin dry-run output (`cache/official_standin_dryrun/out/results.json`). It is NOT a result. Two things in it confirm the wiring: M−ii has identical fake inputs (Laplacian delta on fakes = 0.0), and the label cell is `orig/post/with_firefly`.
- dtype check: `b0_train` and `in_scoring_set` load as bool in 003 `splits.csv` and in the fov splits, so `astype(bool)` is safe.

## Verified correct (no action)
- **Primary contrast and sign.** The primary contrast is AUROC(M) − AUROC(ii) (`run_official.py:291-296`). Both settings use the same RAISE `RealSet` and the same `FakeSet`. The fakes are `clean` in both, so only the reals' encoding differs, and the bootstrap is fully paired: the same `Wr` and `Wf` (`h4.py:97-115`).
- **Direction.** RIGID is "higher = fake" everywhere. It is never flipped: `vec()` only selects the condition per class.
- **Best σ and gate.**
  - Best σ = argmax AUROC(ii) (`h4.py:143-146`).
  - The gate uses the CI lower bound of AUROC(ii) at the best σ, and requires RIGID AUROC(ii) > texture AUROC(ii) (`h4.py:132-140`).
  - The gate is evaluated before the thresholds. The thresholds are unchanged: 0.10 / 0.03, with B0 < 0.05 (`h4.py:149-167`).
  - The label cell is `orig` / `post` / `with_firefly` (`run_official.py:301`).
- **B0 gap.** It comes from the same cell and the same contrast. It uses LOGO models (each generator scored by the model that never saw it), standardized on the b0_train reals, with grouped CV and balanced classes (`run_official.py:144-179`).
- **No test leakage.**
  - Texture direction: calib reals vs train fakes, on `clean` only, with TEST_ONLY excluded (`run_official.py:224-230`).
  - Residualization: calib reals of the pool (`:257-266`).
  - 1% threshold: calib reals (`:350-353`).
  - B0: b0_train only. FLUX test-only is asserted (`:152`, `:602`).
- **Same noise across conditions.** The noise is identical for an image across conditions and arms, because it is seeded by (sha, draw, σ) (`run.py:75-83`). So the M/ii/iii contrasts on the same reals are paired in the noise as well.
- **Processing.** Within each setting, both classes get identical processing. The only exception is the intended format asymmetry of M (and of setting i).
- **Laplacian input.** It is computed on the 224 input after the condition and before the noise (`run_official.py:107-108`).
- **004 `fov_prep.py` diff.** No silent behavior change at the defaults:
  - `size=512`, `mode=fov` and the default out-dir `data/processed/fov512` are unchanged.
  - A user `--out-dir` is now resolved to an absolute path. This only fixes `relative_to(ROOT)` in the report. `rewrite_paths` already resolved it, so the paths in the splits are unchanged.
  - `--mode crop` writes to `data/processed/crop512/images/<sha>.png`. That equals the `path` column of 003 `splits.csv`, which I checked.

## Findings (ranked)

### 1. HIGH (pre-registration completeness; does not change the RIGID label): B0-JM is not implemented anywhere
- **Where:** `docs/04-phase4-plan.md:72`.
- **What is missing:** the pre-registration includes "B0 trained with each train fake JPEG-compressed at q ~ U[85,95] w.p. 0.5", with its own success criterion. That criterion is: the per-real-pool gap AUROC(VISION) − AUROC(RAISE) falls from 0.086 to ≤ 0.03, with a LOGO-macro clean loss ≤ 0.01.
- **Evidence:** `grep -rn "JM"` finds it only in the plan. It is absent from the run script, the config and notes.md.
- **Fix:** before the official data arrives, do one of the following:
  - (a) implement it as an extra B0 arm in `run_official.py`, with the JPEG augmentation on the b0_train fakes only, seeded per sha. The JPEG features need a 004 extraction or an on-the-fly embedding. Report the VISION − RAISE clean gap per arm.
  - (b) log a dated amendment that defers it explicitly.
- **Label impact:** none on the RIGID label. However, it is a pre-registered H4 outcome that would otherwise go silently unreported.

### 2. MEDIUM (could change the label): silent B0 fallbacks are still active in the official (non-`--allow-missing`) mode
- **Fallback (a), a condition can be skipped.** `run_official.py:679-682` catches `FileNotFoundError` from `load_aligned_multi` and only logs "skipped". If the 004 `jpeg(90)` cache is missing:
  - B0 has no M, iii or iii-p values;
  - `b0_gap` becomes NaN;
  - the label cell becomes `NOT_EVALUABLE` (`h4.py:159-160`) instead of aborting.
- **Fallback (b), rows can be dropped from B0 training.** `fit_b0` (`run_official.py:149-151`) silently drops b0_train rows that have no cached features. For example, RAISE train reals or SD3.5 train fakes may be missing because 004 was run before the data was re-fetched. B0 is then trained on a different set, which shifts the B0 gap. That decides SUPPORTED vs INCONCLUSIVE (B0 gap < 0.05).
- **Fallback (c), NaN test rows.** Any NaN in the test rows of B0 makes `macro_auroc` NaN for the whole setting (`h4.py:110-111`).
- **Fix:** when `allow` is False:
  - raise if any condition in `ARMC[arm]` has no cache;
  - raise if any `b0_train` row has NaN features;
  - raise if any active test or calib row has NaN features or NaN RIGID scores.
  - Keep the current tolerant behavior only for `--dry-run` / `--allow-missing`.

### 3. MEDIUM (could change the label): RIGID score caches are keyed only by the list of image shas
- **Where:** `run_official.py:650-654` (also `:664-666`, the Laplacian-only cache).
- **What is wrong:** `cache/official_scores/<arm>__<cond>.npz` is reused when `keys == shas`. Nothing else is checked:
  - not σ or draws (these come from `config.json`, which has been edited before; see the Amendment 6 disclosure);
  - not the feature-variant order, the model, the batch size or the code version.
- **Consequence:** if σ values change but their number stays the same, the old scores are loaded under the new σ labels without any error. Best σ and the gate could then come from the wrong σ.
- **Fix:** store `sigmas`, `draws`, `FVS`, `cond`, `arm`, the model id and a hash of `config.json["rigid"]` in each npz. Assert equality on load, and recompute on any mismatch.

### 4. LOW-MEDIUM (process; D-008): the guard covers only orig / clean, and it passes trivially if the run lands on CPU
- **Where:** `run_official.py:637` → `run.py:150-195`.
- **What is wrong:**
  - The 32-item subset uses VISION native and test fakes in the `orig` arm, `clean` only. RAISE, `jpeg(90)`, `jpeg75_resize05` and the new `fov224` arm are not checked. The infrastructure section of the plan says "for every new condition".
  - If `get_device` returns CPU (for example, CUDA hidden by mistake), the guard compares CPU with CPU and passes.
- **Fix:**
  - In the official mode, assert `dev.type == "cuda"`, or record `device` prominently in tables.md.
  - Extend the subset with about 8 RAISE reals plus 8 fakes under `jpeg(90)`, and a few `fov224` images. This is cheap: about 100 extra forwards.
- **Label impact:** none unless CUDA is numerically off.

### 5. LOW (secondary contrasts only): every `RealSet` uses the same bootstrap seed
- **Where:** `run_official.py:205`, `h4.py:93`.
- **What happens:** the native-device draws and the RAISE-image draws come from identically seeded streams with different K. Contrasts that mix pools, such as (i)−(iii), (i)−(iii-p) and (M)−(iii) on the native side, then get arbitrarily correlated real-side resamples instead of independent ones. The CI width is slightly mis-stated.
- **Fix:** seed each RealSet with `[seed, crc32(pool)]`, like `FakeSet`.
- **Label impact:** none. The primary M−ii contrast uses one pool, so it is correctly fully paired.

### 6. LOW (disclosure; touches the gate's CI): RAISE reals are bootstrapped per image
- **Where:** `h4.py:91`; config `aggregation`.
- **What happens:** RAISE has no device id. `camera_or_platform` is "RAISE" for all 399 rows. Amendment 3 says "bootstrap over devices (reals)". With per-image clusters, the AUROC(ii) CI is narrower than a camera-clustered one would be. That narrower CI lower bound is used by the validity gate.
- **Fix:** keep the current behavior, which is the only option available, and state it in tables.md and notes.md next to the gate.

### 7. LOW (interpretation): REFUTED uses the signed gap
- **Where:** `h4.py:163`.
- **What happens:** a strongly negative M−ii gap (JPEG reals make RIGID worse) is labelled REFUTED. This matches the literal pre-registration and the pilot's `verdict()`. Say so in the write-up, and print the sign of the gap next to the label.

### 8. LOW (CLAUDE.md process): the notes.md step-1 `fov_prep.py --mode crop` overwrites existing files outside `experiments/`
- **Where:** the notes.md command list; `fov_prep.py:97`.
- **What happens:** the default out-dir is `data/processed/crop512`, which holds the existing pilot/003 crops. They are rewritten in place. The bytes should be identical, but CLAUDE.md says to stop and ask before overwriting anything outside `experiments/`. The step also never verifies that a re-created crop equals the one already there.
- **Fix:** run with `--out-dir data/processed/crop512_check`, then compare the sha256 of each PNG against the existing crop where both exist. Then copy only the missing files, after the owner agrees. Also note that the CSV diff in notes.md can only be "identical" if no raw file is missing.

## Not checked (outside CPU-light scope)
- The CUDA path of the guard (still pending, as notes.md says).
- The actual official data: RAISE, SD3.5 and FLUX are not on disk.
