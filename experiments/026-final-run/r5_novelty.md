# R5 novelty check (docs/11), written 2026-10-29 23:3x
Scope: 11 distinct web searches + primary arXiv pages (abs/HTML) for 7 papers. Search is NOT exhaustive; "novel" = "not found". Fetched pages were read through a summarising tool, so quotes are [unverified] against the PDF figures. KB entries added: repsel-arxiv2026, cioni-eccvw2024, xu-origin-wacv2025, imageattributionbench-arxiv2026.

## (a) Mid-depth ViT/CLIP/DINO layers beat the last layer for ATTRIBUTION, also after JPEG/resize
**Verdict: KNOWN qualitatively; partially novel only as a trained-probe, laundering-trained, size-controlled measurement.**
1. Li, Bongini, Tondi, Barni, "Representation and Reference Selection in Training-Free Synthetic Image Attribution", arXiv 2607.12052 (Jul 2026, preprint). Abstract: "attribution accuracy consistently peaks at intermediate representation levels" (CLIP and DINOv2). Table II (CLIP): L12 best everywhere; BC-Attr-6 M=10: 56.67 (L6), 85.33 (L12), 29.25 (L24). Fig. 5: under JPEG/blur/resize "accuracy still peaks at intermediate CLIP layers and drops toward the final layer". Fig. 4: semantic separability grows with depth, generator separability peaks mid-depth. Gap vs us: training-free reference matching, perturbations only at test time, no trained probe, no train-time laundering.
2. Cioni et al., arXiv 2408.09153, ECCV-W 2024. Sec. 4.3 layer sweep: "DINOv2 shows the best Open Set performance in the very first layers" (open-set, favours EARLY layers; partly contradicts a mid-layer story for open-set). Does not claim a closed-set mid>last gap.
3. Xu et al., arXiv 2403.19653: "mid-level" style (VGG Gram) features beat RGB (92.80%, Sec. 5); not ViT layers.
Also (detection, not attribution): RINE (kb/papers/rine-eccv2024.md, Table 4: 89.8 full vs 80.8 last-layer only); arXiv 2605.04358 (intermediate layers for detection, preprint).
Implication: do NOT claim the mid-layer effect as new. Claim only: effect size under laundering with a trained probe, caption-family folds, 8 size-matched generators (+0.254 b8-b12).

## (b) Laundering-augmented attribution probe transfers to UNSEEN laundering types
**Verdict: NOVEL (no instance found); neighbours exist.**
- Cioni 2408.09153 Sec. 4.3: "the methods are immunized by adding the perturbation as augmentation during the training phase" - same perturbation in train and test as far as seen; no held-out type.
- ImageAttributionBench 2605.12967 (Table 3: DS 0.5/0.25, JPEG q65/q30, blur s3/s5): clean-trained attributors drop "from 90.4% to 47.3%"; no held-out-degradation training protocol in the portion read (last 12k chars unread).
- POSE (kb/papers/pose-cvpr2023.md) and Yu et al. 2019 (kb/papers/yu-iccv2019.md): augmentation/fine-tuning restores robustness for the SAME perturbations.
- WILD (kb/papers/wild-ijcnn2025.md, Table I): ViT 95.8 -> 88.67 after 3 steps; processing chains in test, training not laundering-aware.
- Xu 2403.19653: a separate model per perturbation; no transfer test.
- Leave-one-degradation-out exists only for proactive-defence work (PADL 2409.17941, snippet only, not attribution) [unverified].
- Defactify-4 team reports (e.g. 2503.10718) were seen in snippets only [unverified]; challenge entries use augmentation and 2609.11188 trains with compression/resize/blur/gray augmentation, but its abstract reports no held-out-type test.
Caveat: absence in 11 searches; challenge reports and non-arXiv work may contain it.

## (c) Frozen generic foundation features beat real/fake-detector embeddings for attribution
**Verdict: NOVEL as a direct controlled comparison (not found); weakly supported by neighbours.**
- Cioni 2408.09153 and Xu 2403.19653 (Table 5: frozen CLIP+LP 70.15%, DINOv2+LP 67.68%, fine-tuned 95.31/96.67%) show generic frozen features work, but compare to from-scratch/fine-tuned nets, not detector embeddings.
- Li 2607.12052: only CLIP vs DINOv2; no detector embeddings.
- DE-FAKE (kb/papers/defake-ccs2023.md) and RPA (kb/papers/rpa-arxiv2026.md, Table 1: DE-FAKE 62.0/57.3 vs 98.9/95.0) compare attribution heads, not detector-embedding reuse.
- Our result (detector embeddings -0.118 vs generic DINOv2; B-Free / Community Forensics backbones) has no matching published comparison found. Note: one result of ours is from a few fine-tuned backbones, so "detector-trained" is a small sample.

## (d) Patch-shuffle / texture-vs-semantics analysis of attribution features
**Verdict: PARTIALLY KNOWN as a claim ("style/texture, not content"); NOVEL as a patch-shuffle test of ViT-feature attribution.**
- Xu 2403.19653 Sec. 5: generator signatures "might be more intricately tied to their style rather than the direct visual content"; App. C.4 crops: 2x2 patches 22.29% vs 7.69% chance. Sec. 4.1/Table 2 says attribution still "relies on the visible content" in cross-domain tests (semantic dependence, which pulls against a pure-texture claim). No patch shuffle.
- Cioni App. 0.A.5: retrieval accesses "texture, color and focus" (qualitative).
- Li 2607.12052 Fig. 4: mid layers separate generators, final layers separate semantics (probe, no shuffle). Also reports reference semantics matter ("semantically constrained references help").
- Patch shuffle exists only for DETECTION: arXiv 2511.19126 (CLIP, "When Semantics Regulate"), Smash&Reconstruction arXiv 2311.12397 (detection) [snippets only].
- ImageAttributionBench 2605.12967 reports accuracy falls on semantically disjoint splits (90.4 -> about 45.6), i.e. attribution can lean on content - an argument R2 must address.
Implication: R2 can be framed as testing a claim others suggest but nobody measured with tile shuffle on frozen-ViT layer profiles.

## Net for claim C
- Safe to claim: quantified, size-controlled, caption-family-fold layer profile for trained probes under laundering; transfer to held-out launderings; detector-vs-generic comparison.
- Not safe: "mid-layer advantage for attribution is new" (Li 2026 precedes; also RINE for detection).
- Cite Li 2607.12052 and Cioni 2408.09153 in docs/10 sec. 6 (currently says "not reported in the ~30 attribution papers we read"). Both unread-in-PDF; verify before publication.


## Corrections after the final-run critic (lead, 2026-10-31)
- (b) "Transfer to held-out launderings" is no longer a safe claim: the launderings were close to the training mix, and new mild blur or unsharp drop the probe to about 0.52 (exploratory).
- (c) Retracted: CommFor at a mid block beats DINOv2-base (+0.026 [0.006, 0.047]); CLIP-OpenAI vs CommFor +0.014 [-0.007, 0.034].
Full reading: `docs/12-final-run-report.md`, `experiments/026-final-run/critique.md`.
