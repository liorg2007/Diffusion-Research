# Brought a Gun to a Knife Fight: Modern VFM Baselines Outgun Specialized Detectors on In-the-Wild AI Image Detection
- **id:** vfm-baseline-arxiv2025
- **Authors / venue / year:** Yue Zhou, Xinan He, Kaiqing Lin, Bing Fan, Feng Ding, Jinhua Zeng, Bin Li / arXiv preprint only (cs.CV; submitted 16 Sep 2025, v3 15 Oct 2025; no peer-reviewed venue stated) / 2025
- **Links:** arXiv 2509.12995 · https://arxiv.org/abs/2509.12995 · code URL: not found in fetched text -> **[unverified]**
- **Family:** foundation-features
- **Status:** read-full (arXiv v3 HTML via fetch tool)

## Summary (plain language)
A linear classifier on frozen [CLS] features of a modern vision foundation model (DINOv3, Perception Encoder, Meta CLIP-2, SigLIP-2...) trained on one GenImage subset beats specialised detectors by a large margin on in-the-wild sets. The authors suspect part of this is exposure of the foundation model's pre-training data to the test data, so they build a new "unseen" set (WebAIG-25) to check; only DINOv3 stays strong there.

## How it works
- Frozen VFM, linear classifier on [CLS] feature. VFMs: DINOv3, DINOv2 (self-supervised); CLIP, Meta CLIP, Meta CLIP-2, SigLIP, SigLIP-2, PE (language-supervised), "largest officially released weights".
- Probe layer, input resolution, JPEG/resize handling: **not stated in the fetched text** -> **[unverified]**. Text-image cosine similarity probe (Sec. 3.2, Table 5) examines "AI generated"-like concepts.

## Training data
GenImage SD v1.4 subset, no augmentation. Other detectors' numbers copied from their original publications (not re-trained).

## Reported results
- **WebAIG-25 (Table 4, "unseen" set): DINOv3 R.Acc 97.5 / F.Acc 91.4 / Avg 94.5** - the figures in the brief are confirmed. Others (R.Acc/F.Acc/Avg): UnivFD 59.2/29.8/44.5; NPR 93.5/34.9/64.2; SAFE 99.7/21.3/60.5; C2P-CLIP 84.7/31.0/57.9; CLIP 46.7/71.1/58.9; SigLIP2 69.5/55.8/62.7; Meta CLIP-2 27.4/75.4/51.4; PE 44.4/91.7/68.1; DINOv2 40.9/54.2/47.6.
- **Evaluation set**: WebAIG-25 = fakes: r/midjourney posts made after the models' training cut-offs; reals: private campus photographs taken by the authors. **Size and explicit cutoff date not stated** -> **[unverified]**. Confound: real = private campus photos vs fake = Reddit Midjourney posts (different content domain, source, and likely processing), so accuracy partly measures domain/source shortcut; and a single set, no stated threshold.
- Chameleon (Table 1): DINOv3 GenImage 96.3; Chameleon R.Acc 93.7 / F.Acc 90.5 / Avg 92.4 (-3.9 vs GenImage); PE 96.1; Meta CLIP-2 91.8; specialised detectors: AIDE 58.4, UnivFD 57.2, CNNSpot 56.9. (Specialised-detector numbers copied from prior papers.)
- WildRF (Table 2, Avg): DINOv3 96.4 (Facebook 94.7, Reddit 96.7, Twitter 97.9); PE 85.1; SigLIP2 84.7; DINOv2 71.2; UnivFD 55.3; NPR 63.5.
- SocialRF / CommunityAI (Table 3, Avg): DINOv3 93.3 / 95.3; PE 75.1 / 97.1; UnivFD 54.4 / 67.0.
- Abstract claims >20% in-the-wild gain over specialised detectors (not derived in a single table per fetch notes).
- Calibration / FPR: R.Acc and F.Acc reported separately; threshold not stated -> **[unverified]**.

## Known weaknesses / failure modes
- Paper's own conclusions: gains from VLMs shrink on data post-dating pre-training; authors "speculate" data exposure explains much of it (not proven); VLMs show high fake recall but low real accuracy (CLIP 46.7 real acc, Meta CLIP-2 27.4, PE 44.4 on WebAIG-25) = high false positive rate on real photos.
- No dedicated limitations section; only one private unseen set of unstated size; internal inconsistencies in text vs tables (e.g. "all encoders >70% on WildRF" but SigLIP 65.7). Not peer reviewed; baselines not retrained.
- Later papers: none found.

## Code + weights
Not identified. DINOv3 official weights are Meta's (license terms of DINOv3 not checked here) -> **[unverified]**.

## Verified?
partially: all numbers above from the arXiv v3 HTML via a fetch tool (the table header was reported as garbled in Table 1, column assignments inferred); not cross-checked against the PDF.

## Relevance to this project
- Strongest claim that DINOv3 linear probe generalizes to post-cutoff Midjourney content (97.5/91.4), but the real-photo side is a single private campus set, so it does not demonstrate low FPR on diverse phone/social photos.
- Shows modern CLIP-family models are biased toward predicting "fake" on real photos; DINOv3 is the exception here - worth testing FPR on our own fresh real photos.
- Training on SD1.4 GenImage (known JPEG/size bias) with no augmentation and still reaching WildRF 96.4 suggests the probe is not just a format shortcut, but this needs our own check.
