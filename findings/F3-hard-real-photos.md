# F3: polished and DSLR photos false-alarm more than phone photos

**Finding.** Held-out FPR with 5 training sources, jpeg75_resize05:

| source | FPR | AUROC |
|---|---|---|
| Unsplash (polished) | 17.0% | 0.660 |
| RAISE (DSLR) | 14.2% | 0.670 |
| Commons (enthusiast) | 7.0% | 0.738 |
| Open Images (Flickr) | 1.5% | 0.811 |
| VISION (2010s phones) | 1.0% | 0.861 |
| HDR+ (Nexus/Pixel) | 0.5% | 0.863 |

**Why it matters.** It runs against the common focus on phone pipelines as the main risk.

**Evidence.** `experiments/019-source-diversity/results/{tables.md,critique.md}`; `experiments/016-loso/results/critique.md` for the "polished" reading.

**Limits.** Only 6 sources, and **no modern phones**: HDR+ is Nexus and Pixel phones reprocessed with a 2017 pipeline, and VISION is 2010s. "Polished" is a visual reading of a few dozen images, not a measured attribute; no simple image attribute explained it. This must not be read as "modern phones are safe".
