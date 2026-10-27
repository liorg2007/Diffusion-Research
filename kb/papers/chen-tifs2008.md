# Determining Image Origin and Integrity Using Sensor Noise
- **id:** chen-tifs2008
- **Authors / venue / year:** Chen, Fridrich, Goljan, Lukas; IEEE Trans. Information Forensics and Security vol. 3 no. 1 pp. 74-90, March 2008 [citation as given in a Binghamton report via search; the search snippet printed vol. 1 / 2008, volume unverified]
- **Links:** no arXiv · IEEE TIFS · no official code located
- **Family:** camera-forensics
- **Status:** read-abstract-only

## Summary (plain language)
Extends Lukas et al. 2006: a maximum-likelihood style PRNU estimator and a better detector (PCE, peak-to-correlation energy) for deciding camera origin, plus using the missing fingerprint in a region to flag forgeries. It is the reference recipe most later PRNU work (including the VISION benchmark) builds on.

## How it works
Wavelet-denoise residuals -> weighted ML fingerprint estimate from N images (flat or natural) -> normalised correlation / PCE detector with a threshold chosen for a target false-alarm rate. Region-level detection of absent PRNU for tamper localisation.

## Training data
Not read [unverified].

## Reported results
No numbers read [unverified]. Only the bibliographic record and the existence of a forgery-detection use (SPIE 2006 companion paper, via search snippet) were confirmed.

## Known weaknesses / failure modes
Same limits as lukas-tifs2006; also later work shows modern phone pipelines (HDR, portrait mode) create non-unique shared patterns causing false positives (see samsung-prnu-arxiv2025, sdnp-tifs2026).

## Code + weights
No official repo found by us. Binghamton group publishes MATLAB code for PRNU extraction on its site (not checked) [unverified]; patents exist (US8160293B1).

## Verified?
No. Only citation metadata via web search.

## Relevance to this project
- PCE with threshold 60 is the convention that Amped/other follow-ups use; we should use PCE not raw correlation if we run a PRNU control.
- Not a detector of AI images.
