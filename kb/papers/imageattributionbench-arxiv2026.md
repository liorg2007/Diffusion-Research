# ImageAttributionBench: How Far Are We from Generalizable Attribution?
- **id:** imageattributionbench-arxiv2026
- **Authors / venue / year:** Tingshu Mou, Zhipeng Wei, Chao Gong, Jingjing Chen, Xingjun Ma / arXiv preprint (v1 13 May 2026) / 2026
- **Links:** arXiv 2605.12967 · https://arxiv.org/abs/2605.12967
- **Family:** attribution
- **Status:** partially-verified (abstract + first ~100k chars of HTML via fetch summary)

## Summary (plain language)
Benchmark of many generators across semantic domains. Existing attributors collapse on degraded images and on semantically disjoint splits.

## How it works
Evaluates published attributors (incl. a frozen-CLIP + linear one, SSP, PatchCraft, FSD, Omni-DFA) on clean-trained, degraded-tested images. Degradations (Table 3): downsample 0.5x/0.25x, JPEG q65/q30, blur sigma 3/5.

## Reported results
- "average accuracy of all attributors drops sharply from 90.4% to 47.3%" (degraded test); semantic-disjoint split: 90.4% to about 45.6%.
- Training seems clean-only; no held-out-degradation protocol described.

## Known weaknesses / failure modes
No layer analysis, patch shuffle or mid-layer probe seen in the read portion.

## Code + weights
Not checked.

## Verified?
Partially.

## Relevance to this project
- Confirms the clean-train collapse that our T-mix laundering training addresses; useful baseline context. Not a test of train-time laundering transfer.
