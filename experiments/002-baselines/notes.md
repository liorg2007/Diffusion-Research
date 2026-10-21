# 002-baselines: pretrained baseline detectors (set up 2026-10-21, no evaluation run yet)

Usage: `python experiments/lib/run_eval.py --manifest M.csv --out DIR --path experiments/002-baselines --scorer scorers.<module>:<fn>`
(`DD_DEVICE=cpu|mps` overrides the device; default MPS with permanent CPU fallback on RuntimeError). All scorers return
**logits** (float64, higher = more fake; sigmoid = the official probability; official decision threshold = logit 0), lazily load the
model on first call, import the official model/transform code **by path, unmodified** from `third_party/` (no repo setup scripts run).
Details/hashes: `config.json`; provenance rows: `env/SOURCES_baselines_pending.md`; smoke numbers: `smoke_results.json` (`smoke.py`).

| scorer (`scorers.`) | model | license | test-time preprocessing (official) | resizes? |
|---|---|---|---|---|
| `communityforensics:score` | ViT-S/16 @384, OwensLab/commfor-model-384 | MIT | RGB -> Resize(440, shorter side, bilinear) -> CenterCrop(384) -> ImageNet norm | **yes** (512 -> 440 -> crop 384) |
| `bfree:score` | DINOv2-reg4 ViT-B/14, BFREE_dino2reg4 | **NON-COMMERCIAL** | no resize, no pixel crop: whole image patch-embedded (patch 14, floor), five 36x36-token windows (=504 px; centre+4 corners), logits averaged; <504 px tiled; ImageNet norm | no |
| `rine:score` (LDM ckpt), `rine:score_4class` (ProGAN-4class) | frozen OpenAI CLIP ViT-L/14 (fp32) + RINE head | Apache-2.0 (+CLIP MIT) | CenterCrop(224) (zero-pad if smaller) -> CLIP norm. `*_tencrop`: TenCrop(224), mean of 10 logits (paper protocol for Synthbuster/commercial tools) | no |
| `clipdet:score` (latent10k_plus), `clipdet:score_10k` | open_clip ViT-L/14 CommonPool.XL, next-to-last feats, L2-norm, linear | Apache-2.0 (+laion MIT) | Resize(224, bicubic, shorter side) -> CenterCrop(224) -> CLIP norm | **yes** (512 -> 224) |

Preprocessing consequence for robustness runs: CommunityForensics and ClipDet see a down-scaled image (so a prior resize/JPEG interacts with their own
resize); B-Free and RINE look at native-resolution pixels (RINE only a 224 centre patch by default). B-Free on a 512x512 input uses a 36x36 token grid
(drops the last 8 px row/column), and its five windows coincide, so we compute one (5x faster, verified bit-identical on 8 images incl. a 1024x1024 one that uses the
real multi-window path: `bfree_dedupe_check`). Official code is batch-size-1; we batch same-size images (identical results).

## Which checkpoints (choices to revisit)
- Community Forensics: 384 model only (best in paper; 224 model not downloaded).
- RINE: default = **LDM-trained** ckpt (diffusion-trained, far better on diffusion than ProGAN-trained; paper Table 6); ProGAN-4class kept as `score_4class` for the paper's main
  protocol. All four ckpts are in `data/weights/rine/` (1class/2class unused).
- ClipDet: `latent10k_plus` (repo default, includes resized/compressed reference images). Repo default additionally fuses with Corvi2023 (ResNet50) by soft-OR; **not** reproduced (CLIP-only, as in the project's "CLIP few-shot" baseline).

## Smoke results (details in smoke_results.json)
- All scores finite on 7 synthetic images (incl. non-square), MPS vs CPU max |diff| <= 4.2e-5 logit (also on 12 images).
- Official outputs reproduced: Community Forensics notebook probs [0.9988,0.9878,0.9569,0.9516,0.7860] (all 5 match at 4 digits; logit err 0.029 = rounding of published probs);
  **B-Free demo_images/results.csv reproduced to 3e-5 (MPS) / 5e-6 (CPU)**. RINE has no published numbers: demo real jpg p(fake)=0.006, demo fake png ~1.0 (4class). ClipDet ships no
  expected outputs (needs Synthbuster download); sanity only (3 DALL-E-2 fakes: LLR +0.97/-0.15/+1.95, a real photo +0.12 for latent10k_plus -- weak separation on 4 images, not informative; verified the backbone tensors equal the safetensors file).
- Throughput, 512x512 inputs, MPS (AMD 5500M) / CPU img/s: CommunityForensics 36 / 11; B-Free 5.3 / 1.5; RINE 9.3 / 2.4 (tencrop 0.72 / 0.21); ClipDet 9.2 / 2.4.
  (Excludes PNG decode and perturbation time. Estimated full-set cost per condition scales as 1/throughput.)

## Deviations / caveats
1. **MPS silent-error hazard**: a ViT-L forward over 40 images on MPS returned wrong logits (|diff| up to 0.4 vs CPU) *without raising*, so the CPU fallback does not catch it; <=30 images were fine.
   Default batch sizes keep every forward <=20 images (RINE tencrop uses 2 images x 10 crops). The smoke test includes a 12-image MPS-vs-CPU check; re-run `smoke.py` before changing batch sizes.
2. RINE: CLIP kept in fp32 (official `clip.load` is fp16 on GPU; stored weights are fp16, so this is an upcast). `clip.load` replaced by a shim so RINE's own `Model` class uses the local backbone;
   trainable weights loaded with `load_state_dict` instead of the repo's `exec` assignment (same tensors; asserted no unexpected keys).
3. ClipDet: backbone read from `open_clip_pytorch_model.safetensors` (same HF repo as the `.bin` the code fetches; chosen to stay in the size budget) via two temporary monkey-patches around model construction.
4. Community Forensics: model built with `pretrained=False` (the repo's constructor would download ImageNet init weights that the checkpoint overwrites).
5. B-Free: dedupe of identical windows (above) is the only change to the forward pass.
6. `HF_HOME` is set to `data/hf_cache` and `HF_HUB_OFFLINE=1` inside the scorers (nothing is downloaded at score time).
7. Repos are shallow clones (`--depth 1`, LFS smudge skipped; ClipDet heads fetched from the LFS raw URL and checked against the pointer oid). `third_party/CLIP` (openai/CLIP) was cloned in addition to the four baselines because RINE's backbone is OpenAI CLIP; only `clip/model.py` is imported.

## Licenses
- **B-Free: GRIP-UNINA custom licence, NON-COMMERCIAL ("informational and nonprofit purposes"; industrial/profit use prohibited; authors must be credited in published results).** Code and weights alike. Fine for this research project; do not ship it in any product, and cite the authors in any publication.
- Community Forensics MIT (code + HF weights). RINE Apache-2.0. ClipDet Apache-2.0. OpenAI CLIP MIT. laion CLIP-ViT-L-14-CommonPool MIT (HF tag).

## Packages added to env (via `uv add`, torch/numpy untouched: torch 2.2.2, numpy 1.26.4)
`open-clip-torch==2.24.0` (+ transitive new: `ftfy 6.3.1`, `wcwidth 0.9.3`, `protobuf 7.36.2`, `sentencepiece 0.2.2`). `env/pyproject.toml` and `env/uv.lock` updated.

## Downloads
Weights ~3.16 GB (B-Free zip 0.32, CF 0.09, OpenAI CLIP 0.93, laion CommonPool 1.71, RINE ckpts 0.11 via git, ClipDet heads ~10 KB) + shallow git clones ~0.27 GB (RINE clone 224 MB incl. ckpts) = ~3.4 GB; on disk `data/weights` = 3.55 GB (incl. the 346 MB unzipped B-Free copy next to its zip). Under the 4.5 GB weight budget.
