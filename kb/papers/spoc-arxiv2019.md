# SpoC: Spoofing Camera Fingerprints
- **id:** spoc-arxiv2019
- **Authors / venue / year:** Cozzolino, Thies, Rossler, Niessner, Verdoliva; arXiv Nov 2019 (v2 Apr 2021); venue not named on arXiv page
- **Links:** arXiv 1911.12069 · no code link on arXiv page
- **Family:** camera-forensics
- **Status:** read-abstract-only

## Summary (plain language)
A GAN is trained to inject a chosen real camera model's in-camera traces into GAN-generated images so that camera-forensics detectors (and generic GAN-image detectors) call them authentic. The attacker needs only sample images from the target camera, no knowledge of the detector.

## How it works
GAN-based trace spoofing conditioned on target camera model, per abstract.

## Training data
Not read.

## Reported results
Abstract only: works across many conditions; no numbers read [unverified].

## Known weaknesses / failure modes
This is the attack: camera-trace based checks (Noiseprint-like, camera-model ID) can be defeated by synthesis; so 'has camera trace' is weak evidence of realness.

## Code + weights
No code link found.

## Verified?
No. Abstract only.

## Relevance to this project
- Warns against using camera-trace presence as a realness proof; relevant to RAID-style transfer attacks.
- Supports using traces as a feature only with adversarial testing.
