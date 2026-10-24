# A Debiased Reconstruction-based Framework for Training-Free Detection of AI-Generated Images (RDD)
- **id:** rdd-cvpr2026
- **Authors / venue / year:** Sungik Choi, Hankook Lee, Jaehoon Lee, Robin Kim, Stanley Jungkyu Choi, Moontae Lee (LG AI Research and others) / CVPR 2026 (pp. 3254-3263 per search snippet) / 2026
- **Links:** https://openaccess.thecvf.com/content/CVPR2026/html/Choi_A_Debiased_Reconstruction-based_Framework_for_Training-Free_Detection_of_AI-Generated_Images_CVPR_2026_paper.html · arXiv: none found · code: none found
- **Family:** reconstruction
- **Status:** read-abstract-only

## Summary (plain language)
Image-level LDM reconstruction error (AEROBLADE style) is biased by image content, especially simple backgrounds. RDD normalizes the error using augmented copies of the same image (rotation, low-pass filtering) and adds a latent-level reconstruction error with a rotation-based normalization. The two debiased scores are combined into one training-free score.

## How it works
Per the abstract (via chatpaper mirror and search snippet): image-level score = reconstruction error normalized by errors on augmented versions with similar background content; latent-level score = latent reconstruction error debiased by rotation; final RDD = combination. Details (which AEs, exact formulas, passes) NOT read **[unverified]**. The full PDF and supplement returned HTTP 403.
- **Weights needed:** an LDM autoencoder; whether the latent-level score also needs the UNet is not confirmed **[unverified]**.
- **Passes per image:** several AE passes (one per augmentation) at least; exact count **[unverified]**.

## Training data
None (training-free).

## Reported results
- **Unseen-generator results:** abstract only claims SOTA training-free performance "across diverse generative models"; no numbers or generator list read **[unverified]**.
- **Laundering results:** abstract says the framework "can be robust to corruption of the examined images"; which corruptions and numbers **[unverified]**.
- **FPR on real photos:** unknown **[unverified]**.

## Known weaknesses / failure modes
Not read. Reconstruction-based detectors in general are adversarially fragile (Fragile Reconstruction, arXiv 2604.12781).

## Code + weights
- **License:** unknown **[unverified]**; no code link found.

## Verified?
No (abstract-level only). Re-fetch the CVPR PDF from a machine that is not blocked before relying on it.

## Relevance to this project
- Same group as HFI and WaRPAD; the "debias by rotation/low-pass augmentation" idea targets exactly the content-dependence that hurts AEROBLADE on simple real photos (a likely false-positive source on phone photos).
- Highest priority paper to read in full for a VAE-only method.
