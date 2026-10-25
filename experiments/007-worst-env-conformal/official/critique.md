# Critique: OFFICIAL H2-v2 (worst-environment conformal FPR), experiment 007

**Critic:** opus/high, independent. **Date:** 2026-10-25.
**Inputs read:**
- the pre-registration: `docs/04-phase4-plan.md` (H2 section, Amendment 3 item 4, Amendment 4, Amendment 13);
- the summaries: `official/summary_{orig,fov512}.md`;
- the result directories: `orig_*`, `fov512_*` and `orig_Aarms_primary_platform_min100/`;
- the descriptive tables: `raise_descriptive_*.md`;
- the code: `build_scores.py`, `run_analyses.py`, `h2v2.py`, `analyze.py`;
- the D-008 guards: `cache/official/d008_guard_*.json`;
- context: the pilot critique, the LOG tail and D-016.

**Checks** are in `official/critic_checks/`. They are CPU only and read-only.
- `c1_fpr_recompute.py`: an independent re-implementation of the conformal FPR, with Clopper-Pearson (CP) intervals, the α-degeneracy test and the FPR/TPR trade-off.
- `c2_per_device.py`: B0 false positives per held-out device.
- `c3_per_device_Aarms.py`: the same for A0/A1/A2.
- I also re-ran the lead's A-arm analysis into the scratchpad, with both candidate split files.

## Verdict

**NOT_SUPPORTED stands for the primary A0, and for B0 / B0-twinfree / CommFor / RINE / B-Free.** ClipDet is SUPPORTED but irrelevant: it is a weak detector, and its jpeg75_resize05 FPR is 27%.

No bug changes the label. But two points change how the result should be read:
- one tested α value is degenerate;
- the false-positive tail comes from a device, not from a platform.

## 1. Bugs and deviations that could change the label

| Item | Finding | Could it change the label? |
|---|---|---|
| Numbers reproduce | B0 orig at M2 (pooled@1%, LODO×LOPO): 23/1790 = **1.28%**, worst platform WA 9/358 = **2.51%**. Both equal summary_orig.md (c1). The A-arm results are reproduced exactly with `--splits official/splits_orig.csv`: A0 TPR 29.09%, FPR 1.40%, labels identical. With 003 `splits.csv` the labels are also identical (TPR within 0.6 pp). | No |
| 585 test fakes | The train-split fakes are dropped for every detector. This is needed for B0, which saw them in training. Pretrained detectors scored all 720 in the pilot. The verdict compares worst-group vs pooled at matched TPR *within* a detector, so the fake set cannot flip it. | No |
| screenshot_sim as a synthetic platform | Not a deviation: the original H2 text pre-registers both jpeg75_resize05 and screenshot_sim as synthetic platforms. Newly scored rows passed D-008 (max \|Δlogit\| ≤ 1.1e-4). | No |
| B0 TPR is not LOGO | B0 saw 7 of the 8 generators, so its TPR is optimistic: 34.0% clean at a deployable 1% threshold, but only **10.0% on the unseen FLUX** (c1). The label is unaffected: the comparison is within one detector, and the LOGO version (A0) gives the same label. | No |
| A-arm analysis run by the lead | The results are reproduced, but the exact command is not recorded anywhere in the repo. Add it to notes.md (`analyze.py --v2 --scores 005/official/orig/scores_long.csv.gz --splits official/splits_orig.csv --groupings platform --min-group 100`). | No |
| **α = 0.5% is structurally degenerate** (new) | With platform groups of about 344 calibration reals each, `wg@0.5%` equals each group's maximum, so the worst-group threshold equals the overall maximum. That equals `pooled@0.05%` (clipped to the maximum) in **125/125 cells** (c1). So diff ≡ 0 with CI [0, 0] for every detector, and it is always counted as "diff ≥ 0", one of the 3 of 4 votes NOT_SUPPORTED needs. With it excluded, strictly positive diffs at the 3 informative α are: A0 2/3, B0 2/3, A1 3/3, A2 2/3. Under a majority rule the label is unchanged. It would become INCONCLUSIVE only under unanimity, which was never pre-registered. The rule as adopted in Amendment 4 is applied correctly. | No under the adopted rule; disclose it |
| Statistical caveat | CP intervals on 1790 treat cells as independent. They are not: there are 3 versions of each photo, the synthetic cells are transforms of the natives, and errors cluster by device (below). The device-bootstrap CI is the honest one. A0 pooled@1%: 1.40% [0.06%, 3.77%]. | No |

## 2. Is NOT_SUPPORTED the right reading?

Yes, and in the regime that matters it is stronger than the vote count suggests.
- **At the target α = 1%, worst-group is significantly *worse* than pooled at matched TPR for A0:** +0.25 pp [+0.01, +0.60]. It is also worse for B0: +0.10 pp [0.00, +0.15].
- **Why it cannot work here (new):** B0's false positives are not a platform effect. They come from **one phone**:
  - **D17 (Microsoft Lumia 640 LTE)** accounts for 17 of 23 FPs in orig and **30 of 30 in fov512**. In fov512 that is 6/15 = 40% of its photos on every platform (c2).
  - Under A0, D17 carries 19.6 of 25.1 FPs (c3).
  - So the "worst platform = WA 2.51%" is mostly D17's WhatsApp copies: 6 of the 9 WA false positives.
  - A threshold that takes the max over *platform* groups cannot target a device-specific tail.
  - The device is the true environment variable, and it has only about 45 photos per device. That is too few for a per-device conformal threshold (the pilot's M3 degeneracy).
- **Implication for the report:** H2's premise, "platform shift drives the FPR tail", is not what the data show. The tail is phone-specific, which is H3's question.

## 3. What the FPR numbers mean for the Phase-5 FPR ≤ 1% target

**Pooled FPR at the 1% setting.** Conformal calibration at nominal 1% gives:

| Model | Pooled FPR | Worst platform |
|---|---|---|
| B0 | 1.28% | 2.51% |
| A0 | 1.40% | 2.43% |

So the guarantee leaks by about 1.3–2.5× under device and platform shift.

**Per-phone FPR is bimodal:** 0% on 21–24 of the 25 phones, 40% on one. A pooled "≤ 1%" can coexist with an unusable detector for owners of the wrong phone.

**The cost of pushing the threshold down.** Getting every platform to about 1% needs pooled α ≈ 0.25% (WA 1.12%; c1). At that threshold B0 (orig) catches only:
- 21.7% of clean fakes;
- 13.7% of laundered fakes;
- **5.3% of the unseen FLUX** fakes.

**What the data can certify.** With 358 reals per platform, even 0 errors gives a CP upper bound of 1.03%, so this data can never certify ≤ 1%.

**Who the phones are.** VISION's phones are 2013–2016 models without neural ISPs. Whether a modern phone is a "0% phone" or a "D17 phone" is untested until H3 (the owner's photos).

**RAISE**, as a descriptive group, is 1.2% clean and 2.3% laundered at pooled@1% (`raise_descriptive_orig.md`).

**Bottom line:** the ≤ 1% target is not met by any detector here in the per-phone sense. Meeting it on average costs most of the detection power on unseen generators.

## 4. Plain-language takeaway for the owner

We tested a smarter way to set the "this is fake" threshold. Instead of one cut-off tuned on all real photos, we took the strictest cut-off needed by any one platform (camera original, WhatsApp, Facebook). It did not help. At the same detection rate, it gave the same or slightly more false alarms than simply lowering the ordinary threshold.

The deeper finding is why. Almost all false alarms come from a single phone, a 2015 Microsoft Lumia, not from WhatsApp or Facebook. That phone's photos get flagged as fake up to 40% of the time, while 21–24 of the 25 phones in the test set get no false alarms at the 1% setting.

So "1% false alarms on average" hides "0% for most people, 40% for some". Making the detector strict enough to stay near 1% everywhere leaves it catching only about 1 in 20 images from a generator it has never seen.

For the Phase-5 goal (≤ 1% false alarms on fresh phone photos), the deciding test is per phone, on modern phones. That is your own photo set (H3), not more threshold tricks.
