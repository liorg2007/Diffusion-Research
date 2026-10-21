# LAION-Mobile: Evaluating Deepfake Detectors On One Million Smartphone Photos
- **id:** laion-mobile-arxiv2026
- **Authors / venue / year:** Achim von Stryk, Janis Keuper. arXiv preprint, 10 Sep 2026. No venue (not peer-reviewed as far as found).
- **Links:** arXiv 2609.11134 · https://arxiv.org/abs/2609.11134 · data https://huggingface.co/datasets/laionmobile/laion-mobile · code: no repo URL given
- **Family:** robustness-attacks
- **Status:** partially-verified

## Summary (plain language)
The only study found that directly measures false alarms of AI-image detectors on real smartphone photos. Builds a ~1M-image real-phone set (via EXIF) and evaluates 12 detectors. Finds no detector both beats chance on modern AI content and keeps a usable false-alarm rate on real phone photos.

## How it works
Source: re-LAION-5B images >1024 px; EXIF Make in 36 phone manufacturers; dedup; non-photo filter (caption, aspect >1.9, missing exposure EXIF) -> 935,399 IDs from 1,035,208. Eval sample 10,000 requested, 9,171 retrieved, 9,115 scored (DIRE only on 738). Detectors, original checkpoints (Table 4): RIGID-DINOv3, UnivFD, RIGID, AEROBLADE, CLIP-RB, IAPL, DIRE, CNNDetection, Effort, Tiny-LaDeDa, NPR, AIDE.

## Training data
n/a (evaluation paper). Reference AI-content sets: ProGAN (ForenSynths) and NTIRE 2026.

## Reported results
- AUC on NTIRE 2026: 0.42-0.62; best RIGID-DINOv3 0.624 (CI 0.612-0.635); five detectors <0.5 (Table 4).
- FPR at tau=0.5: from 0.0% to 98.1% (DIRE) — uninformative (Sec 5.3).
- FPR at ProGAN EER threshold: UnivFD 0.2%, IAPL 2.1%, Effort 3.6%, CLIP-RB 6.1%, CNNDetection 10.5%. At NTIRE-EER threshold: 17.4% (CNNDetection), 39.1% (UnivFD), 42.9% (CLIP-RB), 89.3% (IAPL), 91.3% (Effort) -> the abstract's "17-91%".
- Era effect (Sec 5.4): predicted rise of FPR for heavier ISP NOT supported; most detectors no significant change, four move opposite.
- Only 212 images (0.03%) from phones released 2021+; pool peaks 2014-2016; heavy-ISP phones 10.9% (Fig 4, Table 1).

## Known weaknesses / failure modes
Authors' limits (Sec 7): real-only so FPR only; "real" inferred from EXIF (forgeable); thresholds transferred from synthetic sets so FPR mixes domain shift; no current flagship phones; single run per pair; source images not redistributed, ~8% URL decay. JPEG handling not described in extracted text [unverified].

## Code + weights
Dataset (IDs, EXIF, hashes, scores; no images) on HF; paper page CC BY 4.0; dataset license not stated in paper (HF tag shows a cc-* license, truncated: unverified).

## Verified?
Partially: arXiv HTML via fetch summary; HF API metadata. Date is after my training knowledge, so treat as read only through the tool summary.

## Relevance to this project
- Directly relevant to phone-photo FPR, but covers old phones; the neural-ISP (2021+) question remains open.
- Argues for reporting FPR on fresh real photos at a threshold fixed on an independent source; supports our own controlled phone capture.
