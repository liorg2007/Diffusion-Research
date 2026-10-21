# RAISE (raw DSLR images)
- **Source/URL:** http://loki.disi.unitn.it/RAISE/ · download http://loki.disi.unitn.it/RAISE/download.html · **Paper:** Dang-Nguyen et al., "RAISE - A Raw Images Dataset for Digital Image Forensics", ACM MMSys 2015 · **License:** "non-commercial research and educational purposes" (download page) -> **NON-COMMERCIAL**.
- **Size:** 8,156 raw images, ~350 GB for RAISE-All (about 43 MB/image, my calculation); subsets RAISE-1k/2k/4k/6k (no size stated); custom download by category and camera (Nikon D90, D7000, D40). Flat-field set ~5 GB. -> RAISE-1k raw is far over 5 GB; <=~100 raw images fit.
- **Real sources:** 4 photographers, 3 Nikon DSLRs, 2011-2014, >80 places in Europe. Not phone photos.
- **Generators:** none; Synthbuster and B-Free-extended fakes were prompted from RAISE-1k.
- **Formats:** RAW (format name and resolution not stated on pages read **[unverified]**). Needs conversion (e.g. to PNG) - the conversion/resizing pipeline becomes part of the real class. Easiest: use the already-converted resized PNG set in bfree-extended-synthbuster.md (1.5 GB).
- **Download procedure:** web form selection; no scripted URL list seen.
- **Known issues:** camera/era mismatch with modern phones; raw-to-PNG conversion differs from phone JPEG pipeline.
- **Used in our experiments?** no
