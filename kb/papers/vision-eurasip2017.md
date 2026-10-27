# VISION: a video and image dataset for source identification
- **id:** vision-eurasip2017
- **Authors / venue / year:** Shullani, Fontani, Iuliani, Al Shaya, Piva; EURASIP Journal on Information Security 2017 (doi 10.1186/s13635-017-0067-2)
- **Links:** https://doi.org/10.1186/s13635-017-0067-2 · dataset https://lesc.dinfo.unifi.it/VISION/ · see also kb/datasets/vision-phone-photos.md
- **Family:** camera-forensics
- **Status:** partially-verified

## Summary (plain language)
VISION is the benchmark of native and social-network-shared photos and videos from 35 phones/tablets, built so that PRNU-style source identification can be tested after Facebook/WhatsApp/YouTube processing. It supplies flat-field (wall/sky) and natural images per device.

## How it works
Dataset paper. Each device has flat-field images, natural images, and copies shared through Facebook (high and low quality), WhatsApp, and YouTube for videos. Shared copies are downscaled/recompressed by the platform.

## Training data
n/a (benchmark). Search summary: 34,427 images and 1914 videos, 35 devices, 11 brands, native plus social versions. Our own count from VISION_files.txt is in kb/datasets/vision-phone-photos.md.

## Reported results
The paper's own PRNU accuracy numbers were NOT readable (Springer PDF blocked behind a cookie redirect) [unverified]. Independent datum from the Amped blog (2020-09-18, not peer reviewed): on VISION, cross-device PCE comparisons showed no outliers well above the threshold of 60 and the few weak false positives stayed below 100 (VISION devices are 2010-2015 era).

## Known weaknesses / failure modes
Old low-res sensors; few photographers; shared copies are downscaled so full-resolution PRNU alignment is lost (our copies: FB long side 2048, WhatsApp 1280, see docs/07). Used as real training data by community forensics (communityforensics-cvpr2025), so not independent of that detector.

## Code + weights
Dataset CC BY-SA 4.0 per README (kb/datasets/vision-phone-photos.md).

## Verified?
Partially: metadata and dataset size from a search summary; licence from our existing KB entry; Amped statement from the blog fetch. Paper results not read.

## Relevance to this project
- Our only real-photo source with per-device ground truth and platform-processed twins, so a natural test bed for any 'real fingerprint' idea.
- Only ~14 native images per device in our subset (500 total, see docs/07): too few for classic PRNU fingerprints.
