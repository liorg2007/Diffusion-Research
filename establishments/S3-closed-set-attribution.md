# S3: known generators can be named after laundering

Linear probe on a mid-level block, 8 generators, jpeg75_resize05 (chance 0.125):

| backbone | balanced accuracy |
|---|---|
| CLIP-OpenAI | 0.896 |
| CLIP-LAION | 0.886 |
| Community Forensics | 0.883 |
| DINOv2-base | 0.856 |
| B-Free | 0.839 |

- **Size shortcut ruled out:** dalle2/sd3/flux, all native 1024², reach 0.964. SD2 vs SDXL, with identical size sets, reach 0.93.
- **Dataset-batch reading not supported:** Synthbuster-only 6-way 0.867; sd3 vs flux 0.960.
- **Replication:** RAID n=500, 4-way 0.772 (chance 0.25); fresh 400 ids 0.770.
- **Training matters:** clean-trained DINOv2 falls to 0.542 on laundered images.

**Biggest open hole.** Each class is one configuration from one operator, so generator and settings cannot be separated.

**Evidence.** `experiments/020-attr-foundation/results/{tables.md,critique.md}`, `.../secondary/tables.md`, `experiments/026-final-run/r1r4/results/tables.md`, `experiments/026-final-run/critique.md`.

**Context.** A published ViT reports 88.67 after processing (wild-ijcnn2025), in a different setup.
