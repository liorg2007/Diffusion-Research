# Test-time laundering canonicalization + a frozen backbone outside the attacker's ensemble as a transfer-attack defense
- **id:** canonicalize-vs-transfer-attack
- **Gap addressed:** G7 (adversarial transfer robustness evaluated separately from laundering and calibration)
- **Scores (1-5):** novelty 2 · expected gain on unseen/in-the-wild 2 · Mac feasibility 4 · **product 16** (rank 5)

- **Hypothesis:** On RAID transfer-attack images at eps = 8/255, the frozen DINOv3 multi-layer probe B0 (never part of RAID's attacked ensemble) loses <= 0.10 AUROC relative to the clean originals, and applying the canonical laundering (JPEG q75 + 0.5x resize) to every input at test time recovers >= 50% of whatever loss remains, at a clean-AUROC cost <= 0.03 - whereas UnivFD (part of the ensemble) loses > 0.30.

- **Rationale:**
  - Transfer attacks crafted on CLIP/CNN detectors move CLIP-like features; a self-supervised DINOv3 feature space with a linear head trained on laundered data is a different target. If our detector is already designed to be laundering-invariant, applying that laundering at test time is a free purification step - its clean cost is measured anyway in laundering-nuisance-projection.
  - The project brief requires robustness to transfer attacks; RAID makes this measurable without running attacks.

- **Closest prior work:**
  - KB: raid-arxiv2025 (transfer from K24: O23 AUROC 0.95 -> 0.50, Ch24CN 0.94 -> 0.59, Table 2; held-out DINOv2 accuracy 83.1 -> 54.6 at eps 16/255, Table 4), mavali-aigi-robustness-arxiv2025 (eps 8/255 attacks survive "Medium" 75% resize + JPEG Q50; RobustCLIP defense costs 3-7 AUC on benign data, Table VI), fpba-tmm2025 (blur+JPEG training no defense, Table VIII), carlini-farid-cvprw2020 (lowest-bit attacks survive JPEG).
  - Web check: Mavali et al. (https://arxiv.org/pdf/2410.01574) is the closest - they show degradation after attack does not neutralise Diverse-Input transfer attacks. JPEG-as-purification is an old adversarial-defense idea in general vision; I found no AIGI paper testing it jointly with a laundering-trained frozen SSL probe and RAID. Novelty is low; value is checking the brief's robustness criterion cheaply.

- **Evidence for / against:**
  - For: RAID Table 4 shows held-out baselines degrade much less than ensemble members (CoDE 91.9 -> 75.6 acc vs ensemble members to ~0.5 AUROC).
  - Against: Mavali shows Diverse-Input attacks survive 75% resize + Q50; DINOv2 (a held-out SSL model) still fell to 54.6 acc at eps 16/255 (RAID Table 4). Expect partial support at best.

- **Test plan (Mac-feasible):**
  - Data: RAID (HF aimagelab/RAID, MIT per HF tags in raid-arxiv2025) - sample ~500 adversarial images at eps 8/255 plus their ~500 clean originals and ~500 reals; check dataset structure/size before download [unverified]. Note RAID generators are SD-family/DeepFloyd IF with D3 reals; report clean AUROC on these first (it is a different distribution).
  - Detectors: B0, B0 + canonicalization, UnivFD, CommFor ViT-S.
  - Metric: AUROC clean vs adversarial, and TPR at the worst-env-conformal-fpr threshold.
  - **Pass:** B0 drop <= 0.10 AUROC; canonicalization recovers >= 50% of the drop; clean cost <= 0.03. **Fail:** B0 drop > 0.20 or recovery < 25%.
  - Optional: own PGD transfer from UnivFD surrogate (eps 8/255, 10 steps) on 100 images - backprop through ViT-L may need CPU; time-box to 2 h.

- **Pilot result (2026-10-24, UNINFORMATIVE; experiments/010-raid-transfer/pilot/, critique.md):**
  - Our B0 probe fails the validity gate on RAID: clean AUROC 0.643. It calls 77% of LAION/D3 web reals fake, because it learned "real = camera photo" from 99 VISION training reals.
  - Out-of-ensemble references at eps 8: CommFor 0.952 → 0.719, ClipDet 0.821 → 0.735.
  - Post-hoc noise controls (not pre-registered): random-sign noise of the same magnitude explains ClipDet's drop fully, and about 0.09–0.17 of CommFor's 0.233. At most ~0.065 is image-specific.
  - The drop comes from attacked fakes; attacked reals barely move.
  - JPEG q75 + 0.5x canonicalization nearly cancels the attack inside canonicalized space (residual drop 0.007–0.061), but costs about as much clean AUROC (0.08–0.16). RINE and ClipDet then flag 35–44% of reals.
  - Conclusion: idea #5 is untested in its proper form (a canonicalized-input detector trained in-domain), not refuted. A proper test needs about 0.9 GB more RAID originals, which exceeds the 9 GB cap and needs the owner.
- **Status:** pilot done (uninformative); official design in the critique
