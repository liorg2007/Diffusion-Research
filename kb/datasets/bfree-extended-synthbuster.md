# B-Free extended Synthbuster (RAISE-1k resized + FLUX + SD3.5 + latent-diffusion)
- **Source/URL:** https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/ · **Paper:** bfree-cvpr2025 · **License:** no separate licence file seen; README says use is under the GRIP-UNINA LICENSE.txt (nonprofit/informational only) -> treat as **NON-COMMERCIAL / research-only** **[licence for data specifically unverified]**.
- **Size (server listing + HTTP Content-Length):** real_RAISE_1k.zip 1,646,333,296 bytes (1.6 GB; listing shows 1.5 GB); sd3_flux.zip 3,436,462,649 bytes (3.4 GB; listing shows 3.2 GB); latent-diffusion.zip 127,295,883 bytes. All three = ~5.2 GB decimal (slightly over a 5 GB budget); real + latent-diffusion = 1.8 GB; or sample sd3_flux via range requests.
- **Real sources:** 1,000 RAISE images "resized to a dimension so that the area is about 1024x1024" (README). Zip contents (my listing): 1,000 PNG.
- **Generators:** FLUX (1,000 PNG), Stable Diffusion 3.5 large (folder sd3_large, 1,000 PNG), latent-diffusion (1,000 PNG); captions extracted from RAISE following the Synthbuster procedure. (Exact FLUX variant not stated **[unverified]**.)
- **Formats:** all PNG for both real and fake -> no JPEG/PNG shortcut; but reals were resized by the authors (interpolation traces) whereas fakes are native. Fake resolution not checked **[unverified]**.
- **Download procedure:** direct HTTPS zip files, server supports range requests (accept-ranges: bytes), so partial extraction works.
- **Known issues:** the real set is resized RAISE (DSLR, 2011-2014, not phones); possible resize-vs-native-resolution shortcut; not peer-reviewed as separate dataset.
- **Used in our experiments?** yes — experiments/001-eval-set (D-005)
