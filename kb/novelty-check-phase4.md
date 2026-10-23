# Novelty check for Phase 4 hypotheses (full-text pass)

Date: 2026-10-23. Phase 2 had only read abstracts. This pass used WebFetch summaries of the HTML full text (a small model extracted the answers, so section numbers are as reported by that tool and are `[unverified]` against the PDFs). Each WebFetch covered only the first ~100k characters of the page. The tail was read for DRIFT only, and it held nothing relevant. Searches were web-search snippets only.

URLs opened:
- https://arxiv.org/html/2606.06918 (DRIFT; also offset 100000)
- https://arxiv.org/html/2603.01544 (RA-Det)
- https://arxiv.org/html/2405.20112 (RIGID)
- https://arxiv.org/html/2602.01973 (calibration paper)
- Search results, not opened in full: https://arxiv.org/pdf/2505.05084, https://arxiv.org/pdf/2512.15182, https://arxiv.org/pdf/2606.09881, https://arxiv.org/pdf/2606.31004, https://arxiv.org/pdf/2508.01248 (NS-Net), https://arxiv.org/pdf/2004.07667 (INLP), https://openaccess.thecvf.com/content/WACV2025W/AI4MFDD/html/Dimitri_Enhancing_Synthetic_Generated-Images_Detection_through_Post-Hoc_Calibration_WACVW_2025_paper.html

## 1. DRIFT (arXiv 2606.06918) vs H1 (laundering-nuisance-projection)

What it does:
- Backbone: frozen DINOv2 ViT-B/14. Only two small MLP heads (robust and fragile) are trained (Sec. 4.2).
- Training data: real images only (MIT-5K, LSUN, RAISE; Sec. 4.1). The loss combines robust-invariance, fragile-drift centering, an ordering margin and a reconstruction anchor (Sec. 3.6, Eq. 11).
- Subspaces are LEARNED by gradient descent. Nothing is closed-form.
- Robust ("physical") transforms: mild JPEG, mild blur, photometric jitter, resampling round-trips (Sec. 3.3; Table 4c).
- Fragile ("edit-like") transforms: strong blur, heavy compression, pixelation, local edits (Sec. 3.4; Table 4d).
- Severity levels are not given. No real platform pairs (WhatsApp, Facebook) are mentioned.
- Held-out transform evaluation: only a sentence in Sec. 4.4 saying leave-one-out "confirms" no reliance on one family. No numbers.
- Metrics are ACC and AP (Tables 1-3). Benchmarks are ForenSynths, Diffusion-6cls and PromptWorld-1K.
- No phone photos and no FPR at a fixed threshold. The decision rule is S(x) > 0 (Sec. 3.7).
- The paper has internal inconsistencies (768-d global feature vs patch-wise method; student/teacher roles differ between Sec. 3.2 and 3.7).

Overlap with H1: PARTIAL, conceptual only.
- Both estimate an invariance/fragility structure from real-only images under transformations on a frozen DINOv2.
- DRIFT's logic is a different one: it treats mild JPEG/resampling as "robust" (the detector should be invariant to it) and treats heavy compression as "fragile" (the detector should respond to it). It is trained, uses synthetic transforms, and tests none of the platform or held-out-laundering questions.

What remains novel for us:
- A closed-form projection (eigendecomposition of the real-only difference covariance followed by I - UU^T), applied before a supervised probe.
- Real platform pairs (VISION native vs WA/FBH) in the estimate.
- Held-out laundering types (`mavali_medium`, `screenshot_sim`) never used in fitting.
- Threshold-drift endpoints and phone-photo FPR.
- We must cite DRIFT as the closest prior work and not claim "first real-only nuisance subspace".

## 2. RA-Det (2603.01544) and RIGID (2405.20112) vs H4 (drift-format-shortcut)

RA-Det:
- Frozen DINOv3 ViT-L/16 with a conditional UNet "Differential Robustness Probing" module (Sec. 3.2.1). It builds discrepancy features (distance, difference vector, cosine; Sec. 3.2.2) and a multi-branch head including a median-filter residual CNN (Sec. 3.2.3).
- It is TRAINED, on 360k real and 360k ProGAN images, and tested on 15+ unseen generators plus SDXL, FLUX.1, MJ v6, Infinity and VAR (Sec. 5.1.1, App. A).
- Training JPEG and blur augmentations are disabled (Sec. 5.1.2). This is a training-time choice, not a control for format.
- The file formats of the real and fake images are never stated.
- Sec. 5.5 tests JPEG QF 95/90/85 and blur at test time. This is a corruption test, not format-matching.
- App. C discusses the PNG-fake vs JPEG-real bias for DIRE only.
- Average 93.47 ACC / 97.00 AP (Table 1).
- Verdict: no format-matched evaluation. Overlap with H4: none on the format question. The method is a trained model and is not RIGID-like.

RIGID:
- Training-free, DINOv2 ViT-L/14, noise lambda = 0.05 (App. A). The threshold is set so that ~95% of reals are correct (Sec. 3.1).
- Data: ImageNet and LSUN-Bedroom fakes from Stein et al., plus GenImage subsets (Sec. 4.1, App. C). File formats are not stated for RIGID's evaluation.
- The format issue appears ONLY for DIRE (Sec. 4.2.1, Tables 1-2 footnote 1, App. D). There the authors found DIRE's gains came from JPEG reals vs PNG fakes and converted both classes to JPEG to retest DIRE.
- JPEG-converted results are reported only for DIRE. None are reported for RIGID or AEROBLADE.
- Sec. 4.3 / Fig. 4 test JPEG q90-50 as a robustness corruption.
- Verdict: RIGID's authors knew about the format shortcut and did NOT check their own method against it. This supports the premise of H4: the gap is real.

Overlap: low to partial. The RIGID authors flagged the artifact for a competitor but did not control for it in RIGID. No paper found runs RIGID-style scores with format-matched classes.

What remains novel for us: the explicit mismatched-vs-matched comparison (settings i-iv) for RIGID-style drift scores with DINOv2, and the B0-JM fix for the supervised probe. H4 is a diagnostic. The novelty is modest, and the finding is publishable only if the gap is large.

## 3. Conformal / worst-group thresholds (H2)

2602.01973 (AAAI 2026, "Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated"):
- Method: a single additive logit bias alpha is added to a frozen detector (Sec. 4, Prop. 2, Eq. 8).
- Supervised variant: a KDE on a small labeled target set; alpha minimizes FPR + FNR, which is the sum of the two class error rates (Eqs. 9-11).
- Unsupervised variant: alpha is chosen by a symmetry criterion on unlabeled target logits (Eqs. 12-13).
- Default validation size is 100 images (Sec. 5.1).
- It does NOT control FPR at a target level and does not mention conformal prediction. It assumes the real distribution is stable (Sec. 3.2).
- Overlap with H2: partial. It confirms threshold shift is the problem, but it is a different objective that needs labeled fakes or a bimodality assumption.

Other searches (4 queries total across items 3-4):
- 2505.05084 (text detection): conformal FPR bound on human text, with power loss (secondhand; the paper itself was not opened).
- 2512.15182 (Hashmi et al., MBZUAI): "safety" threshold for authentic data, built on a resynthesis authenticity index, with no conformal guarantee and no held-out-device evaluation visible in the excerpts. Partial conceptual overlap on "bounded FPR on authentic content". `[unverified]` beyond the snippet.
- 2606.09881: per-group thresholds maximizing worst-group ACCURACY for demographic fairness. This is group-conditional thresholding, but on demographics and with accuracy as the objective.
- 2606.31004 (video audit): argues for reporting at low FPR (0.1%).
- WACV 2025 W (Dimitri et al.): post-hoc calibration across generators.
- No paper found that uses group-conditional / worst-group / Mondrian conformal thresholds for image-detector FPR on held-out cameras or platforms. The search tool returned only snippets, so absence is not proof.

Overlap with H2: PARTIAL, as the idea is incremental. Worst-group quantile and Mondrian conformal are standard tools, and group-wise thresholds exist for fairness. What remains novel for us is the application: real-only calibration with leave-device-out and leave-platform-out evaluation of FPR on held-out phones and platform transforms, with the TPR cost reported.

## 4. Nuisance projection / LEACE / INLP for AI-image detection (H1)

- Search found no application of LEACE or INLP to erase JPEG or resize directions in AI-image detection.
- Nearest: NS-Net (2508.01248) uses a null-space projection to decouple CLIP SEMANTIC content from the features. It targets semantics and not compression. It is trained and not real-only paired. INLP (2004.07667) is the generic method. Neither paper was read in full.
- Overlap: low on application, but the method family (nullspace/erasure) is known. NS-Net should be cited as a related null-space use.
- Remains novel: nuisance = laundering (JPEG, resize, platform), estimated from real paired differences. Caveat: the covariance-of-differences eigenprojection is simple PCA-style erasure, so the novelty lies in the application and the evaluation, not the math.

## Verdicts

H1 (nuisance projection):
- Overlap with DRIFT is partial (learned heads, synthetic transforms, no platform pairs, no FPR). Closed-form real-only projection with real WA/FBH pairs and held-out laundering is not found in the literature.
- Novelty is moderate. Cite DRIFT and NS-Net as related work, and pitch it as a cheap, closed-form, evaluable alternative, not as the first real-only invariance subspace.
- Keep it in Phase 4 as planned. The A1-vs-A2 comparison is the right control.

H2 (worst-env conformal FPR):
- No image-detection paper found with group-conditional conformal on held-out devices and platforms. 2602.01973 does not control FPR and needs labeled fakes or a bimodality assumption.
- Novelty is low to moderate (standard tools, new setting). The value is the empirical protocol. Report it as an evaluation-plus-recipe contribution.
- Keep the plan, and cite 2505.05084, 2606.09881 and 2602.01973.

H4 (drift format shortcut):
- Neither RIGID nor RA-Det controls compression history for its own method. RIGID checked it only for DIRE (App. D). RA-Det disables JPEG augmentation and never states formats.
- The gap is confirmed. Note RA-Det is trained and not RIGID-like, so test the RIGID-style score as planned, and treat RA-Det only as context.
- Novelty is moderate as a diagnostic. Keep it, since it is cheap.

## Findings that should change the plan (also see the report)
1. Use DRIFT's own training transform split (mild = invariant, heavy = fragile) as a related-work contrast. Do not cite DRIFT as claiming real platform pairs.
2. RIGID's authors already show the JPEG-vs-PNG fix for DIRE (App. D). H4 should cite it as precedent and use exactly its protocol (convert both classes to JPEG) for setting (iii).
3. DRIFT is on DINOv2 ViT-B/14, the same backbone family as our B0. If feasible, add a "DRIFT-lite" arm to Phase 4 as a baseline for H1: it is the nearest competitor. This is a suggestion; it needs the owner's or lead's decision.
