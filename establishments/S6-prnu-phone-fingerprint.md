# S6: sensor noise identifies the phone, on 2010s phones

PRNU/PCE, 14 same-size VISION devices, 13–14 reference photos per device, leave-one-photo-out: top-1 **0.985** [0.969, 1.000] (chance 0.071), within-model 1.000. WhatsApp→WhatsApp 0.930; Facebook→Facebook 0.871; native fingerprint→WhatsApp 0.915. Texture baseline 0.346. FPR at PCE 60: bound ≤ 0.0012.

**A replication** of Lukáš 2006 and Chen 2008, not a discovery. The mild surprise is that 13–14 natural photos are enough.

**Limit.** Modern phones are untested, and literature reports that their processing breaks PRNU (Samsung default-mode FPR 0.66–1.0; kb/papers/samsung-prnu-arxiv2025.md).

**Evidence.** `experiments/021-attr-residual/results/{tables.md,critique.md}`.
