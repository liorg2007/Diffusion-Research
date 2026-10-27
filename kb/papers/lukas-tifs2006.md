# Digital Camera Identification from Sensor Pattern Noise
- **id:** lukas-tifs2006
- **Authors / venue / year:** Lukas, Fridrich, Goljan; IEEE Trans. Information Forensics and Security, 2006 (DOI 10.1109/TIFS.2006.873602)
- **Links:** no arXiv · DOI 10.1109/TIFS.2006.873602 · no official code found
- **Family:** camera-forensics
- **Status:** read-abstract-only

## Summary (plain language)
Every camera sensor has tiny per-pixel sensitivity differences (PRNU) that act like a fixed fingerprint. Average the noise residuals of many photos from one camera to get a reference pattern, then correlate a query photo's noise with it to decide whether that camera took it. This is the founding paper of device-level source identification.

## How it works
Denoising filter -> noise residual per image -> average over many images = camera reference pattern -> correlation detector (treated like a spread-spectrum watermark) between query residual and reference. No learning.

## Training data
About 320 images from nine consumer cameras for estimating false-alarm and false-rejection rates (search-result paraphrase of the abstract) [unverified: not read from the paper itself].

## Reported results
Abstract (secondary summary): studies effect of JPEG compression and gamma correction on error rates. Exact error rates NOT read by us [unverified]. A later paper (Matthews et al., via search snippet) notes lens artefacts and temperature were not fully modelled.

## Known weaknesses / failure modes
Needs the reference pattern from many images of the same camera (flat-field or natural) and, in the original setup, same-resolution images; rescaling/cropping breaks pixel alignment (general PRNU fact, see docs/07). Fingerprint can be manipulated/substituted (Li et al. 2010, via snippet).

## Code + weights
No official code located by us. Patents (US7787030B2 etc.) exist on the method - check before any commercial use [unverified scope].

## Verified?
No. Only bibliographic data and abstract paraphrase seen via web search; paper PDF not opened.

## Relevance to this project
- Defines the PRNU baseline we would use on VISION for device-level fingerprints.
- A real-image fingerprint is a *sensor* trace, not a generator trace; its link to AI-image detection is only indirect (see marra-gan-fingerprints-arxiv2018).
