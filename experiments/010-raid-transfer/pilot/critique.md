# Independent critique: E3 PILOT (RAID transfer attacks vs canonicalization, idea #5)

**Rating: UNINFORMATIVE. The label is correct and no bug was found.** The reference-detector story needs two corrections:
- ClipDet's "attack" drop is generic noise sensitivity, not transfer.
- Canonicalization removes most of the attack effect, but it removes as much clean signal along with it. The pre-registered "recovery" ratio hides this.

Critic: opus/high, 2026-10-24.
- Inputs: `pilot/{results.json,scores_long.csv.gz,d008_guard.json,run.log}`, `run.py`, `config.json`, Amendment 9 and both clarifications.
- Scratch scripts and outputs: `pilot/critic_checks/` (`c1_out.txt`, `c2_out.txt`, `c4_out.txt`).
- One post-hoc CPU control was run (c3, below). It is **not pre-registered**. It is a diagnostic only.

## What I verified (no action needed)
- **AUROCs.** I recomputed all 48 pooled AUROCs (4 detectors × 6 sets × 2 analysis sets) from `scores_long.csv.gz` with a rank formula. The maximum difference from `results.json` is 0 (`c1_out.txt` §1).
- **Primary set.** There are 0 padded fakes and 52 padded reals, so the primary set is 48 reals vs 400 fakes, as reported.
- **Pairing.** The worst per-item mean |eps8 − clean| is 7.24 grey levels (limit 10). The worst at eps16 is 10.96 (limit 18). The maximum pixel difference is 9 / 17, which is ε + 1 for rounding (`results.json.pairing_check`). Ids are aligned across classes and sets: I asserted this on the long table in c2.
- **D-008.** All 24/24 detector × set entries passed. The worst max |Δlogit| is 1.0e-4.
- **Independent harness check.** My CPU harness (c3) re-scored 8 clean items per detector and matched the saved CUDA scores to ≤ 4.8e-5 for all four detectors (`c3.log`).
- **Code version.** The run was made from an uncommitted working tree (`git_head` d0ef641). The committed `run.py` (74ee53f) gives the same preprocessing hash, p79d04831, as the caches.
- **Bootstrap.** Each real id is its own cluster, and fakes are resampled per generator. The same plan is used for every set and detector, so drop, gain and cost are paired. The B0 label share is UNINFORMATIVE in 1.00 of the replicates.
- **Per-detector preprocessing** follows the 002 scorers:
  - CommFor: Resize(440) then crop 384, so it upsamples the 256 input 1.7×.
  - ClipDet: resize to 224.
  - RINE and B0: 224 center crop, no resize.

## Ranked findings

### 1. The B0 label is correct and robust. The cause is a collapse of the real class, not a bug (Q1)
- **The gate fails on every view.**
  - Primary clean AUROC is 0.643 [0.560, 0.723]. Even the upper CI bound is below 0.80.
  - Sensitivity (all 100 reals): 0.673.
  - Content-paired win rate, P(fake(id) > real(id)): 0.708 (`c4_out.txt`).
  - Every generator is between 0.598 and 0.687.
- **B0 scores the RAID reals as fake.** 77% of the unpadded RAID reals get a logit above 0, with a median of +4.29 (`c2_out.txt`, `c1_out.txt` §2). For comparison, in 005 `pilot_v2/orig` A0/full:
  - held-out VISION test reals have a median of −4.96, with 13% above 0;
  - Synthbuster test fakes have a median of +9.08.
  The RAID fakes do score slightly lower than in training (medians 5.5–6.8), but the reals moved much further, by about 9 logits.
- **Interpretation [inference].** The probe was trained on only 99 VISION phone-camera reals, so it learned "real = camera photo". RAID/D3 reals are LAION web images: small (unpadded short side median 408 px), JPEG q≈75, often edited. They fall on the fake side.
- **The score does not track real image size** (Spearman −0.20, p = 0.17).
- **The score does track content.**
  - B0's clean scores for real(id) and gen_k(id) correlate with Spearman 0.50 / 0.39 / 0.34 / 0.19 across generators, versus −0.04 to 0.10 for CommFor.
  - So DINOv2-probe scores carry the shared caption/content of each id (the caption pairing itself is [unverified]).
  - The B0 probe is a semantic/domain classifier here, not an artifact detector.
- **Ruled out as bugs:**
  - Direction: the AUROC is above 0.5.
  - Probe: the refit reproduces 005 to 8.3e-4.
  - Inputs: the pairing check passes.
  - Device: D-008 passes, and CPU matches CUDA.
  - Sizes and layout: B0 sees 224 crops of 256 px images. That is a stated domain shift (the 004 training images were 224 crops of 512), not a code error.

### 2. Only part of the out-of-ensemble "transfer" is transfer. ClipDet's drop is entirely generic noise sensitivity (Q2)
Post-hoc control c3 (`c4_out.txt`) uses the primary set and builds two perturbations from the real RAID delta d:
- **signflip**: clean + random-sign |d|. Same per-pixel magnitude, L1, L2 and L∞ as the attack, with the structure destroyed.
- **shuffled**: clean + the d of another id of the same class.

AUROC drop by perturbation:

| Detector | eps8 attack | signflip | shuffled | Reading |
|---|---|---|---|---|
| ClipDet | 0.086 | 0.088 [0.036, 0.143] | 0.062 | Random noise fully reproduces it. Same signature: reals pushed toward fake (median +0.81 attack vs +0.74 noise) and fakes also *up*. |
| CommFor | 0.233 | 0.089 | 0.168 | Image-specific part ≤ about 0.065. |
| B0 | 0.083 | 0.030 | 0.063 | — |
| RINE | 0.144 | 0.033 | 0.057 | Noise pushes everything toward fake; only the real attack evades fakes. Consistent with RINE being in the ensemble. |

- **ClipDet:** "attack reals only" gives 0.629, while "attack fakes only" gives 0.896, which is *above* clean. The fake-side attack does not transfer to ClipDet at all.
- **CommFor:** most of the drop comes from a class-conditional, near-universal pattern plus noise sensitivity. Image-specific transfer is a minority of it.
- **B0 at eps 8 vs eps 16:** the drop is flat (AUROC 0.559 vs 0.573), as it is for ClipDet (0.735 vs 0.724). That fits saturation by a generic perturbation, not alignment with the attack gradient [inference].

### 3. The attack drop is driven by fakes, and the canonicalization cost has a different cause per detector (Q2)
Cross-class decomposition on the primary set, with the paired bootstrap (`c2_out.txt`). Each entry is AUROC(clean) − AUROC(only one class perturbed).

**Attack on fakes vs attack on reals:**

| Detector | Fakes attacked only | Reals attacked only |
|---|---|---|
| CommFor | 0.196 [0.145, 0.246] | 0.007 [−0.008, 0.021] |
| RINE | 0.182 | −0.001 |
| B0 | 0.167 | −0.076 (B0 reals move toward *real*: median −1.18, 81% of them) |
| ClipDet | −0.075 | 0.192 |

- RAID also attacks the reals (toward fake). That half of the attack does **not** transfer to CommFor or B0.
- At a threshold set at 6.2% FPR on clean reals (3/48), CommFor's TPR falls 0.795 → 0.207 while FPR stays at 0.062 (`c1_out.txt` §4). This is pure evasion of fakes.

**Canonicalization clean cost:**
- **CommFor:** 0.108 of its 0.155 cost comes from fakes. Canonicalization alone moves CommFor fake medians by −3.0 to −8.8 logits, about as much as the attack does.
- **RINE:** the cost comes from reals: 0.134 [0.087, 0.185] from reals, −0.049 from fakes. At a threshold with 6.2% clean FPR, the FPR on canonicalized clean reals is 0.354.
- **ClipDet:** the cost comes from reals: 0.389 from reals, −0.136 from fakes. The canonicalized FPR is 0.438.
- So canonicalization destroys the **calibration** of RINE and ClipDet, not only their ranking. Without re-calibration it is unusable against the FPR ≤ 1% target.

**The upsample-back step (clarification 1) does not explain the cost.**
- CommFor and ClipDet resize internally anyway: 440 and 224. From a 128 px bottleneck the information is the same whether or not it is first upsampled to 256.
- The loss comes from the ×0.5 bottleneck plus JPEG. Which of the two dominates is untested. The originals are already q≈75, which points to the resize [inference].

**The recovery ratio is structurally capped.**
- Recovery ≤ [AUROC(clean_canon) − AUROC(eps8)] / drop. On the primary set this cap is 0.336 [0.06, 0.55] for CommFor and 0.437 for RINE. For ClipDet it is −0.05, because clean_canon AUROC is already below the eps8 AUROC (0.730 vs 0.735).
- The 0.5 threshold was therefore out of reach for CommFor, and almost certainly for the others, whatever the purification does.
- The residual drop *inside* canonicalized space, AUROC(clean_canon) − AUROC(eps8_canon), is small:

| Detector | Residual drop in canonicalized space |
|---|---|
| B0 | 0.007 [−0.022, 0.036] |
| RINE | 0.032 [−0.003, 0.067] |
| ClipDet | 0.039 [−0.011, 0.093] |
| CommFor | 0.061 [0.021, 0.102] |

- Correct summary: canonicalization neutralizes most of the eps-8 perturbation. Off-the-shelf detectors lose as much clean signal under canonicalization as the attack removes. The idea is untested in its proper form: a detector trained and calibrated on canonicalized inputs.

**Per generator** (CIs in `results.json`): the CommFor drop is consistent across generators (gen_0 0.839 → 0.561; gen_1..3 0.99 → 0.73–0.80). gen_0, whose originals are native 256 px at about 10 KB, is the weakest clean.

### 4. Sample size: recovery is not identified, and the sensitivity set agrees only for CommFor (Q3)
- **CI widths on the primary set (48 reals):**
  - clean AUROC: 0.163 (B0), 0.044 (CommFor), 0.071 (RINE), 0.121 (ClipDet);
  - drop: about 0.12 for the reference detectors;
  - recovery: 0.67 (CommFor), 0.97 (RINE, B0), 2.49 (ClipDet).
- With 48 reals, "1% FPR" is 0 or 1 image, so no fixed-low-FPR claim is possible.
- **Sensitivity set (all 100 reals):**
  - Agrees for CommFor: drop 0.265, recovery 0.045, cost 0.179.
  - Diverges for RINE: recovery 0.48 vs 0.22. Its drop is 0.200, and the reals-attacked-only component is +0.025.
  - Diverges for B0: recovery 0.58 vs 0.42, and cost 0.010 vs 0.041.
  - The padded reals behave differently (B0 clean median +2.57 vs +4.29), so the black-border shortcut contaminates the sensitivity set, as clarification 2 expected. Do not quote the sensitivity numbers as corroboration, except for CommFor.

### 5. Minor (no change to any number)
- **The bootstrap resamples classes independently even though they share ids.** Content is correlated across classes (finding 1), so a joint by-id resample would give somewhat different CIs. The direction of that difference is untested. Pre-register the joint by-id resample as a sensitivity analysis.
- **The RAID Table 4 anchor is not comparable.** RAID's DINOv2 head is trained on D3 (in-domain); our probe is not. B0's balanced accuracy at logit 0 is 0.585 clean vs RAID's 83.1% accuracy, so there is no basis for an anchor.
- **Attacked reals are attacked toward "fake".** RAID reals are web images, not phone photos, so nothing here bears on the phone-FPR target.

## What this means for idea #5 and the Phase-5 "robust to transfer attacks" target (Q5)
- **Idea #5 is neither supported nor refuted.**
  - The pre-registered B0 is not competent on RAID clean data.
  - Off-the-shelf detectors show that test-time canonicalization trades the attack loss for an equal clean loss.
  - Canonicalized space is nearly attack-invariant (finding 3). That is the half of the idea that looks promising.
- **Phase 5 cannot claim "robust to RAID transfer"** from this pilot. A fair claim needs:
  - an in-domain-competent detector;
  - operating points at fixed FPR;
  - a noise control, to show that the effect is adversarial transfer rather than noise.

## Recommendations for an official E3 (all can be pre-registered before any new result)
1. **In-domain B0.** Train the same A0 recipe (and the variant under test) on **clean RAID/D3 originals with ids disjoint from the test ids**. Never use attacked images in training. Two arms:
   - (a) raw;
   - (b) **canon-trained**: features of canonicalized clean training images.
   Keep the validity gate: clean AUROC ≥ 0.80 with the CI lower bound > 0.5.
   - Download: RAID originals are about 0.9 GB for all 24,000 (`kb/datasets/raid.md`). A few thousand ids are well within D-007, but the lead must check the running total against the 9 GB cap.
2. **Primary metric: net robustness of the deployed pipeline.**
   - AUROC(eps8 → canon, canon-trained arm) compared against AUROC(clean, raw arm), with a pre-set margin, e.g. ≥ −0.05.
   - Also report the residual drop in canonicalized space.
   - Drop the recovery ratio from the label: it is capped by cost/drop and unstable when the drop is small. Keep it descriptive only.
3. **Fixed-FPR operating points.** Set the threshold on held-out **clean** in-domain reals, using the same pipeline as at test (canonicalized if canonicalization is deployed). Then report FPR on attacked reals and TPR on attacked fakes at 5% and 1%.
   - This needs at least about 500 calibration reals for a meaningful 1% FPR. Use separate calibration and test reals.
4. **Noise controls in the protocol.** Add signflip(|d|) and shuffled-d (the c3 definitions). Report the attack-specific drop = AUROC(signflip) − AUROC(eps8). A transfer claim needs that quantity's CI to exclude 0.
5. **Sample size.** Use at least 300 unpadded real test ids, selected by reading the original sizes before sampling. This would narrow the B0 clean CI from about 0.16 to roughly 0.07. That is a √n scaling [estimate].
   - Bootstrap jointly by id across classes, as a sensitivity analysis.
   - Report the content-paired win rate next to AUROC.
6. **Canonicalization ablation, descriptive.** Run resize-only (×0.5 and back) and JPEG-only q75, so the cost can be attributed. For the resizing detectors, also feed the 128 px image directly (alternative D1).
7. **Keep RINE marked as near-white-box.** Base any out-of-ensemble claim only on CommFor or ClipDet after the noise control (item 4).
