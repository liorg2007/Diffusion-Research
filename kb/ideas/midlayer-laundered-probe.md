# Mid-level DINOv2 probe trained on laundered copies (E5 -> E6 -> E7)
- **Origin:** E5 (experiments/014-statistical-field/) found that the real-vs-diffusion difference survives JPEG+resize in DINOv2 block-8 features (content-controlled twin-paired about 0.93, critic).
- **E6 (experiments/015-midlayer-probe/):** P1 PASS. Block-8 + laundered training beats B0 by +0.092 macro AUROC at jpeg75_resize05 (0.947 vs 0.855); FLUX +0.077; held-out launderings +0.07. P2 FAIL: single-source cross transfer 0.644 / 0.604. P3 FAIL: RAISE FPR 7.1%.
- **E7 (experiments/016-loso/):** KILL. Leave-one-real-source-out, fov512, jpeg75_resize05: VISION 0.917 / 0.6% FPR, RAISE 0.689 / 19.1%, Open Images 0.809 / 16.0%. Block 8 beats block 12 (+0.057) and more training sources help (+0.032), but not enough.
- **Mechanism (critic):** the probe learns the real-source domain. Polished photos (bokeh, telephoto, HDR) from an unseen pipeline score as fake.
- **Status:** closed (D-016: one limit raise used; KILL). Negative result.
