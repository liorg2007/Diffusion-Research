# Apple's Synthetic Defocus Noise Pattern: Characterization and Forensic Applications
- **id:** sdnp-tifs2026
- **Authors / venue / year:** Vazquez-Padin, Perez-Gonzalez, Perez-Miguelez; IEEE TIFS vol. 21 pp. 1096-1111, 2026 (arXiv 2505.07380 v2)
- **Links:** arXiv 2505.07380 · CC BY-NC-ND 4.0 (arXiv page)
- **Family:** camera-forensics
- **Status:** partially-verified

## Summary (plain language)
iPhone portrait-mode photos carry a synthetic noise pattern in out-of-focus regions (SDNP) that is shared across devices and can make PRNU verification return false matches. Masking SDNP regions reduces false positives, and the pattern itself can identify iPhone model and iOS version.

## How it works
Estimate SDNP base patterns from 2L portrait images (L=110 in the example), model its dependence on brightness and ISO, mask affected regions before PRNU matching.

## Training data
Dataset (Sec. V): 36 iPhone models, 99 devices, iOS 10 to iOS 26, 2,298 portrait images (Flickr, GSMArena, contest, 3 in-house phones). Resolutions 12MP 4032x3024 and 7MP 3088x2316.

## Reported results
Abstract: masking significantly reduces PRNU false positives; the quantitative results (Sec. VII) were not read [unverified]. PRNU reference uses L native-resolution images, flat-field recommended; minimum L not stated in the part read.

## Known weaknesses / failure modes
Front-camera 7MP patterns left for future work; only portrait-mode images.

## Code + weights
No code seen.

## Verified?
Partially: abstract and Sec. V read via WebFetch (first 100k of 175k characters); results section not read.

## Relevance to this project
- Computational photography leaves device-independent but model/OS-specific traces: a 'neural ISP' signature that could mimic generator fingerprints.
- Portrait-mode phone photos are a risky class for FPR on fresh iPhones.
