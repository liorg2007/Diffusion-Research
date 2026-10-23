# Pre-run code review: E3 PILOT (`run.py`, `config.json`, `notes.md`) against Amendment 9 and its clarification, and `kb/datasets/raid.md`

**Verdict: NOT READY.** The clean control is not the same image as the attacked one for 4 of the 5 classes. I verified this on all 500 items. Because `drop`, `recovery` and the validity gate all depend on AUROC(clean), the B0 label would be invalid. A second problem must be disclosed or handled before the run: RAID's own geometry gives a black-border shortcut to 52% of the reals.

Reviewer: independent pre-run code review, 2026-10-24. CPU only. No model forwards and no scoring were run.
Checks that ran:
- `run.py --stage selftest`: OK.
- Pixel-level pairing checks on the 1,500 manifest files. The scratch scripts are in `/tmp/claude-1000/-home-zera-project-Diffusion-Research/8e17439b-23a0-499c-acb6-22c219043654/scratchpad/review/raid_pair*.py` and `/tmp/claude-1000/-home-zera-project-Diffusion-Research/8e17439b-23a0-499c-acb6-22c219043654/scratchpad/review/raid_checks.py`.

## Verified correct (no action)
- **Id pairing.** Path, id, class and kind are consistent for all 1,500 manifest rows (0 mismatches). The same 100 ids appear in every class, which is asserted (`run.py:110-114`). Items are class-major in the same id order, and `stage_export` asserts this.
- **Bootstrap unit = id** (`run.py:345-365`). One id draw is applied to the reals and to every generator's fakes, the same for every input set and detector. So drop, gain and cost are paired, as pre-registered. The pooled `np.tile(W, (1, 4))` matches the order in which the fakes are concatenated.
- **Label table** (`run.py:334-342`):
  - UNINFORMATIVE if AUROC(clean) < 0.80;
  - PASS if drop ≤ 0.10 and (drop < 0.02 or recovery ≥ 0.5);
  - FAIL if drop > 0.10 and recovery < 0.5;
  - INCONCLUSIVE otherwise.
  It uses the pooled AUROC. Drop = clean − eps8, and recovery = (eps8_canon − eps8) / drop. All of this matches Amendment 9.
- **Canonicalization** = `perturb.jpeg75_resize05` → bicubic back up to 256 (`run.py:155-158`). This matches the clarification. It is applied to both classes, and to clean and attacked inputs alike.
- **D-008.**
  - On CUDA: every detector × input set is checked against a CPU subprocess reference, with abort if > 1e-3 (`run.py:297-321`).
  - TF32 is off on CUDA (`:92-95`).
  - B0 aborts if the backbone falls back to CPU (`:224-225`).
- **Direction.** All 002 scorers return logits with higher = fake (their module docstrings say so). B0 uses `decision_function`. The B0 probe is the A0 recipe, trained on all 459 pilot b0_train rows. `fit_probe` standardizes inside the model, so this is equivalent to 005's pre-standardization; the 8.3e-4 reproduction difference is float-level.
- **Smoke isolation.** The smoke caches live only in `cache/smoke/`. The main `cache/` currently holds no score files, so nothing stale exists yet.

## Findings (ranked)

### 1. BLOCKER (changes the label): the clean control does not reproduce RAID's pre-attack image
- **Where:** `run.py:143-152` (`fov_to_256`) and `:161-166`. The same claim appears in config `input_sets.clean`, notes D2, and the Amendment 9 "Inputs" text, which marked RAID's preprocessing [unverified].
- **What RAID actually does:** the attacked PNG is a plain center crop of the original JPEG to 256, with no resize. When the image is smaller than 256, it is zero-padded first, which is torchvision `CenterCrop(256)` semantics. Over all 500 items, comparing that crop with RAID's files gives:
  - max |eps8 − crop| = 9/255 for every class (attack ε 8 plus 1 for rounding);
  - max |eps16 − crop| = 17/255.
- **What the code does:** it resizes the short side to 256, then center-crops. Measured as the mean |eps8 − ours| per item, in grey levels:

  | Class | Median | Max |
  |---|---|---|
  | real | 63.6 | 205.8 |
  | gen_1 | 67.9 | 115.5 |
  | gen_2 | 67.5 | 115.6 |
  | gen_3 | 83.0 | 125.3 |
  | gen_0 | 6.8 | (matches only because its originals are exactly 256) |

  So for 4 of the 5 classes, "clean" shows a different field of view (zoomed out by 1.4–2×, or upsampled up to 3.7× for small reals). Then:
  - "drop" measures geometry plus attack, not the attack;
  - the gate AUROC(clean) ≥ 0.80 is evaluated on different images.
- **Fix:**
  1. Replace `fov_to_256` with a torchvision-style center crop to 256: no resize; zero-pad symmetrically when smaller (the pad split is `ph//2`, then the rest); crop offsets `round((h−256)/2)`.
  2. Add an asserted pairing check to `stage_verify` for all 500 items: max |eps8 − clean| ≤ 10/255 and max |eps16 − clean| ≤ 18/255.
  3. Update config.json, notes D2, `kb/datasets/raid.md` and Amendment 9 with a dated "clean control now verified" note. Do this before any result.
- Notes D2 and the KB are also factually wrong. They say "most 324", but the real originals have a median short side of 250 (70 to 1,444 px). They say "a few small originals are upsampled", but 52 of the 100 reals are smaller than 256.

### 2. HIGH (bias toward PASS; disclose and pre-register a handling before any result): RAID's geometry gives reals a black-border shortcut
- **What happens:** 52 of the 100 real ids have a short side below 256 (minimum 70 px). RAID zero-pads them, so their eps8 and eps16 PNGs carry black borders. No fake has borders: gen_0 is exactly 256 and gen_1 to gen_3 are 360–512. After fix #1, the clean set has the same borders.
- **Why it matters:** B0's 224 center crop keeps most of each border. A detector can call "real" from the borders whatever the attack does. That keeps AUROC(eps8) high, makes "drop" look small, and pushes the label toward PASS or "drop < 0.02".
- **Fix:** decide and log before any result.
  - Either keep the pre-registered label on all ids, and add a pre-declared sensitivity analysis on the 48 unpadded real ids (all fakes kept, bootstrap still by id). Report the label in both.
  - Or exclude padded ids from the label and say so in the amendment.
  - Either way, report the share of padded reals in tables.md.

### 3. MEDIUM (silent reuse and mixing): score and reference caches are keyed only by item key
- **Where:** `run.py:260-272` (`load_scores` / `save_scores`), `:280-281` (`cpuref_<det>.json`), and `:296` (`d008_guard.json`).
- **What goes wrong:**
  - **(a) Stale scores after fix #1.** Scores in `scores_<det>_clean*.npz` and the `cpuref_*` "clean" entries computed with the old preprocessing would be reused without warning.
  - **(b) Mixed devices and a trivial D-008 guard.** Suppose a partial or full `--device cpu` run is followed by `--device cuda` in the same `cache/`. The guard then compares cached CPU scores with the CPU reference and passes trivially, and the results mix devices.
- **Fix:**
  - Store `device` and a preprocessing-version string (for example, a hash of `make_input` / `canon` source plus SIZE) in every npz and json.
  - Refuse to reuse a cache on any mismatch. On CUDA, always rescore the guard subset.
  - Delete any clean/clean_canon caches that exist after fix #1. Today only `cache/smoke/` has them.

### 4. LOW: the B0 reproduction check only warns
- **Where:** `run.py:196-201`.
- **What happens:** a probe that differs from 005's A0/full by 1e-3 or more is pickled and used anyway. The current difference is 8.3e-4, close to the tolerance.
- **Fix:** raise SystemExit if the difference is 1e-3 or more, or if `REF005` is missing in a non-smoke run.

### 5. INFO (stated caveats, no action)
- B0 sees 256→224 crops, a domain shift.
- The originals are JPEG q≈75, so the canonicalization's clean cost is understated.
- The gen_k-to-name mapping and the upstream licences are [unverified].
- The recovery CI uses only bootstrap replicates with drop > 0.02, and it is descriptive.
- Balanced accuracy at logit 0 is valid, because all scorers return logits.
