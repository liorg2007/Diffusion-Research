# S2: detection is not solved on unseen real sources

No method met the Phase 5 targets or the relaxed DISCOVERY bar (AUROC ≥ 0.90 and FPR ≤ 5% on every held-out source).

| attempt | best honest result on unseen sources | label |
|---|---|---|
| E7, 2 training sources | RAISE 0.689 / 19.1%; OI 0.809 / 16.0% | KILL |
| E8, remove source subspace | no change | KILL |
| E9, adapt with real anchors | RAISE 0.877 / 4.5%; OI 0.893 / 8.3% | KILL |
| E10, 5 training sources | mean 0.767 / 6.9%; worst 0.660 / 17.0% | KILL |

The modern-phone acceptance test (owner photos) and a proper RAID transfer-attack test were **not run**.

**Evidence.** `experiments/016-loso`, `017-embedding-geometry`, `018-real-anchor`, `019-source-diversity`, each `results/`; `docs/03-final-report.md`.
