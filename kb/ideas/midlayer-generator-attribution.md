# Mid-level DINOv2 features attribute the generator after laundering
- **Hypothesis:** a linear probe on DINOv2-base mid-level features, trained on laundered copies, names which of K known generators made an image after JPEG+resize.
- **Rationale:** E5/E6 showed that mid-level features keep a laundering-surviving diffusion signal; attribution asks the same of generator identity.
- **Closest prior work:** wild-ijcnn2025 (ViT 88.67 after 3 processing steps), rpa-arxiv2026, defake-ccs2023. Differences here: frozen backbone, layer contrast, size-matched controls, and unseen launderings.
- **Evidence for / against:**
  - For: 8-way bacc 0.855 [0.834, 0.876] at jpeg75_resize05 (chance 0.125); unseen launderings 0.81–0.87; size-matched dalle2/sd3/flux 0.964; RAID n=500 4-way 0.772; b8 − b12 +0.254.
  - Against: clean-train → laundered 0.542; SD1.3 vs SD1.4 at chance; open-set AUROC 0.768.
  - Sources: experiments/020-attr-foundation/results/{tables.md, critique.md}, secondary/tables.md.
- **Test plan:** done (H-A1, docs/09).
- **Status:** supported (closed-set only).
