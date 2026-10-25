# Open Images V7 (CC-BY Flickr photos, subset)
- **Source/URL:** https://storage.googleapis.com/openimages/web/index.html · download page https://storage.googleapis.com/openimages/web/download_v7.html · **License:** annotations CC BY 4.0 (Google LLC); images listed as CC BY 2.0 but "Google makes no representations or warranties regarding the license status of each image" - check each image's License field in metadata (factsfigures_v7 page).
- **Size:** 9,011,219 train / 41,620 val / 125,436 test images (factsfigures_v7). GB size of val/test not given on the page read.
- **Real sources:** Flickr photos of mixed provenance (many DSLR, some phones); not a phone-only set.
- **Generators:** none (real-only).
- **Formats:** JPEG, as on Flickr (page does not say explicitly that all are JPEG); metadata gives OriginalURL and an optional ~300K-pixel thumbnail (~640x480; generated on the fly and may change) - do not use thumbnails as reals (resize shortcut).
- **Download procedure:** subsets by ID list (`$SPLIT/$IMAGE_ID` lines) with `downloader.py`, or FiftyOne zoo, or CVDF archives. Filter the ID list to License == CC BY before downloading.
- **Known issues:** per-image licence heterogeneity; mixed JPEG qualities and prior platform processing (Flickr resizing); Open Images/OpenImages-derived data may be in generators' training sets (LAION overlap) - possible leakage.
- **Used in our experiments?** no (candidate)

- **Update (2026-10-25):** in the validation-split metadata CSV (2018_04), all 41,620 rows carry License https://creativecommons.org/licenses/by/2.0/, so no per-image filtering was needed for that split. The CVDF bucket copies are re-encoded and resized (short side ≤ 1024; 0/600 match OriginalMD5), so treat them as resized web photos.
