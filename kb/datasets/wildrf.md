# WildRF
- **Source/URL:** Google Drive link in https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld (https://drive.google.com/file/d/1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn/view) · **Paper:** wildrf-ladeda-arxiv2024 · **License:** none found (project page licence applies to the website only: CC BY-SA 4.0; repo LICENSE not read) -> treat as research-only, scraped social-media images **[unverified]**.
- **Size:** Reddit 2,150+2,150; X 340+340; Facebook 160+160 = 5,300 images (paper Fig. 4); GB size not stated.
- **Real sources:** images posted on Reddit/X/Facebook found via authenticity hashtags. **Generators:** unspecified/mixed (#midjourney, #stablediffusion, #deepfake hashtags); labels from hashtags.
- **Formats:** "varying resolutions, formats, compressions, editing transformations" (paper); no PNG/JPEG breakdown. Folder layout train/val/test{reddit,twitter,facebook}/{0_real,1_fake}.
- **Download procedure:** single zip from Google Drive (size unknown, small); no subset mechanism needed.
- **Known issues:** hashtag labels may be noisy; includes drawings/CG per a later critique (unread); tiny Facebook/X test sets; platform-processed so JPEG recompression is present for both classes (good), but fakes vs reals may differ by origin.
- **Used in our experiments?** no (candidate in-the-wild sanity set)
