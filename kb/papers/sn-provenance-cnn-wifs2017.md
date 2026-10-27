# Social network identification through image classification with CNN (platform-provenance of images)
- **id:** sn-provenance-cnn-wifs2017
- **Authors / venue / year:** Authors not verified (Uniroma1/UniFI group per search results: Amerini, Caldelli et al.) [unverified]; WIFS 2017 per the micc.unifi.it filename wifs2017socialnetworkCNN.pdf [unverified]; later journal version IEEE Access listing at research.uniroma1.it
- **Links:** https://www.micc.unifi.it/wp-content/uploads/2018/01/wifs2017socialnetworkCNN.pdf (not parsed) · https://research.uniroma1.it/en/node/54405
- **Family:** camera-forensics
- **Status:** read-abstract-only

## Summary (plain language)
Social platforms (Facebook, WhatsApp, Telegram, Twitter...) recompress and resize uploads in platform-specific ways. A CNN on image content/low-level traces can recognise which platform an image went through, even if metadata is stripped. This is platform provenance, a trace of the pipeline rather than the camera.

## How it works
CNN classifier over the shared image; detects whether an image is from a social network, messaging app or camera (search summary of abstract). Related: FusionNET (Amerini, Li, Caldelli) fuses a 1D-CNN on DCT histograms with a 2D-CNN on sensor-noise residuals; 2021 Pattern Recognition Letters paper uses DCT and wavelet traces on VISION, IPLAB, FODB [all unverified, titles only].

## Training data
Three public datasets from seven social networks (search summary) [unverified].

## Reported results
Abstract claims state-of-the-art; no accuracy numbers read [unverified]. 2021 follow-up reports results on VISION, no numbers read.

## Known weaknesses / failure modes
Not read in detail. Obvious concern: the classifier detects the platform's recompression, which is also what laundering of AI images does; so platform labels could confound real/fake detectors.

## Code + weights
No code located.

## Verified?
No. Search-result summaries only; micc PDF fetch returned binary.

## Relevance to this project
- Platform signature is a confound for our 'social_sim' condition: a detector may learn the platform, not realness.
- VISION natWA/natFBH give free platform labels for real images; a cheap diagnostic: can our detector's score predict platform?
