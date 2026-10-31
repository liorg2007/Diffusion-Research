# Are CLIP features all you need for Universal Synthetic Image Origin Attribution?
- **id:** cioni-eccvw2024
- **Authors / venue / year:** Dario Cioni, Christos Tzelepis, Lorenzo Seidenari, Ioannis Patras / ECCV 2024 Workshops (TWYN), pp. 363-382 / 2024
- **Links:** arXiv 2408.09153 · https://arxiv.org/abs/2408.09153 · code public (URL not checked)
- **Family:** attribution
- **Status:** partially-verified (abstract + HTML via fetch summary)

## Summary (plain language)
Linear/lightweight classifiers on frozen CLIP and DINOv2 features do closed- and open-set origin attribution across GANs and diffusion models, with little training data and better cross-architecture generalization than fingerprint methods.

## How it works
Frozen foundation backbones, features taken from a chosen block (every block swept, Sec. 4.3, Fig. 4), simple classifier plus open-set scoring.

## Training data
GenImage-style generators (details not verified).

## Reported results
- Sec. 4.3: "DINOv2 shows the best Open Set performance in the very first layers, and it sharply decreases in successive layers" (open-set metric, not closed-set accuracy under JPEG).
- Sec. 4.3 / Fig. 6-7: perturbations added "as augmentation during the training phase"; "JPEG compression causes the least drop"; Gaussian noise and blur hurt most. Test perturbation = train perturbation as far as the summary shows **[unverified]**.
- App. 0.A.5: retrieval shows access to "low-level information ... texture, color and focus".

## Known weaknesses / failure modes
No patch shuffle; no detector-embedding comparison; no held-out laundering; layer finding favours EARLY layers in open-set, which differs from our mid-layer closed-set finding.

## Code + weights
Public code per abstract; license not checked.

## Verified?
Partially (see Status).

## Relevance to this project
- Prior art for frozen CLIP/DINOv2 attribution with a layer sweep and augmentation-trained robustness. Does not claim the laundering-trained probe transfers to unseen launderings.
