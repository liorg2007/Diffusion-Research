# 001-eval-set: notes (D-005 evaluation image set)

Date 2026-10-21. Code: `fetch.py` (plan / run / finalize), `config.json`. Outputs: `data/manifest_raw.csv` (3,220 rows; `path` is relative to the project root), `results.json`, resumable state in `state/` (plan.json, done.jsonl, budget.json, run.log).
Selection: `random.Random(0)` freshly seeded per source over name-sorted candidates (VISION: 10 randomly chosen devices get 15 natives, the other 25 get 14, total 500).

## Counts (all downloaded, 0 failed, 0 CRC failures)
| source | group | n | format | resolution |
|---|---|---|---|---|
| VISION native | 35 devices x 14-15 | 500 | JPEG | 960x720 .. 5312x4608, 18 distinct sizes; 3264x2448 (188), 2560x1920 (43), 4640x3480 (30); median 8.0 MP; JPEG q-table[0][0] 1-4 (q~95+); median 2.7 MB |
| VISION WhatsApp copy | same 500 | 500 | JPEG | max side 1280 (1280x960 307x), median 1.2 MP, q-table[0][0]=3 (494/500), median 314 KB |
| VISION Facebook-HQ copy | same 500 | 500 | JPEG | max side 2048 (2048x1536 302x), median 3.1 MP, q-table[0][0] 5-9, median 336 KB |
| RAISE-1k resized (B-Free) | real | 400 | PNG | 5 sizes, 833..1258 px, ~1.05 MP (1258x833 237x); median 1.6 MB |
| Synthbuster dalle2 | fake | 120 | PNG | all 1024x1024 |
| Synthbuster dalle3 | fake | 120 | PNG | 1024x1024 (100), 1792x1024 (13), 1024x1792 (7) |
| Synthbuster firefly | fake | 120 | **JPEG (named .png)** | 1792..2688 x 1536..2304 (4 sizes), median 4.1 MP, q-table[0][0]=2 (all 120, ~q95) |
| Synthbuster midjourney-v5 | fake | 120 | PNG | 4 sizes 896..1360 (1360x896 72x), ~1.2 MP |
| Synthbuster stable-diffusion-xl | fake | 120 | PNG | 26 distinct sizes 576..1728, median ~1 MP |
| Synthbuster stable-diffusion-2 | fake | 120 | PNG | 26 distinct sizes 576..1728, median ~1 MP |
| B-Free sd3_large (SD3.5) | fake | 150 | PNG | all 1024x1024 |
| B-Free flux | fake, split=test_heldout | 150 | PNG | all 1024x1024 |
| B-Free latent-diffusion | fake | 300 (of 1000) | PNG | all 256x256 (!) |

Totals: label 0 = 1,900 (1,500 VISION incl. shared copies + 400 RAISE), label 1 = 1,320. Licences: CC BY-SA 4.0 x1500, GRIP non-commercial x1000 (RAISE-resized reals 400 + SD3.5 150 + FLUX 150 + LDM 300), CC BY-NC-SA 4.0 x720.

## Bytes
Downloaded (all HTTP body bytes incl. central directories and VISION_files.txt): 4,566,674,983 B = 4.567 GB (hard cap 4.6 GB, not hit). By source: vision 1.898 GB, raise_bfree 0.660, synthbuster 1.361, bfree_gen 0.648 (includes the 127.3 MB latent-diffusion.zip). On-disk dataset 4.608 GB (zip PNGs were stored ~uncompressed; the latent-diffusion zip was deleted after verification, only the 300 kept images remain). Slightly above the 4.3 GB estimate because the 120 Synthbuster/ RAISE PNGs are larger than the earlier mean and each ranged entry over-reads ~0.3 KB of padding.

## Integrity
- Range-fetched zips (real_RAISE_1k, synthbuster, sd3_flux): every one of the 1,420 entries decompressed and its size + CRC32 checked against the zip central directory: 0 mismatches. All images additionally fully decoded with PIL.
- latent-diffusion.zip fetched whole: MD5 c8139f0e5a61ff97beaa16ccdab11579 matches the published checksum.txt; SHA256 57c25625843d897ba4305cbe785db317fac1a39feda022e6e08cde085e230317; all 1000 entries CRC32-checked; 300 sampled and kept.
- VISION has no published checksums: files verified by Content-Length (HEAD) equality, and full JPEG decode.
- Zip entry names were sanitised (no absolute / `..` components; output = `<generator folder>/<basename>`). sd3_flux.zip also contains 3 `.ipynb_checkpoints` entries, ignored. Zip central directories are cached in `data/raw/<src>/_central_directory/`.
- Server behaviour: VISION server returned HTTP 503 under 8 concurrent HEADs; we used 3 (HEAD) / 4 (GET) workers, no failures.

## VISION WhatsApp / Facebook mapping (verified)
Mapping is exactly 1:1 by index: `Dxx_I_nat_NNNN.jpg` <-> `Dxx_I_natWA_NNNN.jpg` <-> `Dxx_I_natFBH_NNNN.jpg` (same device dir, same 4-digit index; every device has equal counts of nat/natWA/natFBH). Verified for all 500 triplets: shared copies are downscaled (never upscaled), EXIF stripped, aspect ratio equals the native after applying the native's EXIF orientation (50-55 natives carry EXIF orientation 6/8 and the shared copies have the rotation baked into pixels; 0 unexplained). So no shared-by-device fallback was needed. Pairing is recoverable from the path (device + index); `camera_or_platform` = `Dxx_native|whatsapp|facebook_hq` (underscore-joined).

## Format-bias facts
- Reals VISION are JPEG (native q~95+, WA q~90-ish with max side 1280, FB-HQ max side 2048); fakes are almost all PNG; RAISE reals are PNG (resized by the B-Free authors, ~1 MP) = the format-matched control.
- **Synthbuster firefly files are JPEG data stored under a .png name** (q-table[0][0]=2 ~ q95, 1792-2688 px). Anything that trusts the extension is wrong; this is the only fake generator carrying JPEG artifacts (a JPEG-aware detector gets a "free" cue on it, and it is the one fake that matches VISION in format).
- Resolution is a strong generator-specific/ source-specific cue: LDM 256x256 only, SD3.5/FLUX exactly 1024x1024, DALL-E 2 exactly 1024x1024, RAISE reals ~1.05 MP non-square (833-1258), VISION 8 MP typical native vs 1.2 MP WA vs 3.1 MP FB. Resize to a common size / crop before any model is run, and report results per resolution-matched condition.
- Fakes are native-resolution (no resize); RAISE reals were resized by the authors (interpolation traces) - the resize-vs-native shortcut flagged in the KB remains.
