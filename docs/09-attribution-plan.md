# Attribution pre-registration (D-018 cycle)

**Registered:** 2026-10-27 14:16, before any attribution feature, score or result exists. **Basis:** docs/08-attribution-hypotheses.md (opus/high), which is adopted as written: its sub-questions, methods, data, metrics, PASS/KILL thresholds and validity gates (§3) are frozen by this file. Where this file and docs/08 differ, this file wins.

## Lead decisions
1. **Counting hypotheses** for the D-018 stop rule: H-A1, H-A2, H-A3, H-A4, H-A5.
   - **H-A6** (VAE signature) is a pre-declared, NON-COUNTING diagnostic, since it is predicted to fail.
   - A hypothesis "fails" if its pre-registered label is KILL, or INCONCLUSIVE with the point estimate on the KILL side.
   - **Stop** after 3 counting failures unless the owner raises the limit (D-016 practice).
2. **Label scope.**
   - Each hypothesis's label comes only from its primary criteria in docs/08. Everything else is descriptive.
   - H-A2 passes if (i) or (ii) passes, as in docs/08.
   - DISCOVERY (D-017) applies only where docs/08 defines it (H-A4).
3. **Layer choice in H-A1.** The b4/b8/b12/concat feature set used for the label is chosen by inner grouped CV on training folds only. All are reported, with Holm correction across the 4.
4. **Validity gates.** All six gates in docs/08 §3 are mandatory. A run that fails a gate (M0 not at chance on preprocessed tensors, permutation control off chance, D-008 fail, duplicate sha256 across classes) gets the label INVALID: no label, and it is fixed before any rerun.
5. **Tier-1 downloads** (about 0.4 GB, within the 20 GB ceiling):
   - Synthbuster SD1.3, SD1.4 and GLIDE, 300 each. The licence is CC BY-NC-SA 4.0, the same dataset and licence already approved in the Phase 3 plan.
   - RAID: 400 more ids of originals (MIT card).
   - Used only in pre-declared secondary arms:
     - **Q2-near-twin:** SD1.3 vs SD1.4 pairwise attribution. Descriptive, with chance at 0.5.
     - **Q3-family:** do SD1.x images map to the SD-family decoder (H-A6), and to SD2 rather than to non-SD generators (H-A1 probe)? Descriptive.
     - **RAID replication** at n = 500 ids for H-A1.
   - The main K = 8 set is unchanged. Tier 2 is not approved now.
6. **Experiment folders:**
   - experiments/020-attr-foundation: M0, M1, H-A1, H-A4, all sharing the cached features;
   - 021-attr-residual: H-A2;
   - 022-attr-platform: H-A3;
   - 023-attr-detector-emb: H-A5;
   - 024-attr-vae: H-A6.
   - Each folder holds config.json, run.py, selftest.py, notes.md (with the gate checklist ticked), results/ and critique.md (opus/high).
7. **Code freeze.**
   - Each run's code and its selftest are committed before the real-data evaluation is read.
   - After that, changes are allowed only for bugs. Each change is logged in notes.md with the reason, and both the before and after outputs are kept.
