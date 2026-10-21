# GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image
- **id:** genimage-neurips2023
- **Authors / venue / year:** Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang. NeurIPS 2023 Datasets and Benchmarks track (venue confirmed via neurips.cc poster page in search results; arXiv page lists no venue).
- **Links:** arXiv 2306.08571 · https://arxiv.org/abs/2306.08571 · https://neurips.cc/virtual/2023/poster/73644 · data/code https://github.com/GenImage-Dataset/GenImage
- **Family:** data-bias-benchmarks
- **Status:** partially-verified

## Summary (plain language)
GenImage pairs ImageNet real photos with fakes from 8 generators (Midjourney, SD 1.4/1.5, Wukong, ADM, GLIDE, VQDM, BigGAN) and defines two tasks: train on one generator and test on others, and test under degradations. It became the most common diffusion-detection benchmark, but later work (Grommelt et al.) showed it contains JPEG/size shortcuts.

## How it works
Dataset/benchmark, no new detector. Folder structure per generator: train/val x ai/nature.

## Training data
Real: ImageNet (1,331,167 real images = 1,281,167 train + 50,000 test; Sec 2.1/Table 1). Fake: 1,350,000 (1,300 train + 50 test per class x 1,000 classes). Per generator mostly 162 train / 6 test per class; SD v1.5 166 / 8. Resolutions (Sec 2): MJ V5 1024x1024; Wukong, SD1.4, SD1.5 512x512; GLIDE, VQDM 256x256; BigGAN 128x128; ADM not stated. File format not stated in the paper text read.

## Reported results
- Table 3 (ResNet-50 trained on SD v1.4, accuracy %): Midjourney 54.9, SD1.4 99.9, SD1.5 99.7, ADM 53.5, GLIDE 61.9, Wukong 98.2, VQDM 56.6, BigGAN 52.0, avg 72.1 -> strong in-distribution, near-chance on non-SD-family generators.
- Table 5 (degradations on SD v1.4 test, LR 112/64, JPEG q65/q30, blur sigma 3/5): the extracted column values were inconsistent with the stated average, so I do **not** report them **[unverified]**.
- No FPR at fixed threshold / fresh-real calibration.

## Known weaknesses / failure modes
- Grommelt et al. (arXiv 2403.17608; see fakeorjpeg-arxiv2024): real=JPEG (mostly quality 96) vs fake=PNG, and real ImageNet sizes cluster around 450-550 px vs fixed generator sizes -> detectors can use format/size.
- B-Free (bfree-cvpr2025) builds a "GenImage unbiased" subset (5k real / 5k fake) by compressing fakes at a JPEG quality close to the reals.

## Code + weights
Repo GenImage-Dataset/GenImage: its "License" file states CC BY-NC-SA 4.0 with additional dataset terms, non-commercial academic use only, no selling data or derivatives for gain (read from raw GitHub file). Data distribution: Baidu Yunpan (code ztf1) and a Google Drive mirror (added 2024-03-15), compressed archives; total size not given. No official HF mirror; third-party HF mirrors exist but were not verified as official.

## Verified?
Table 3 and dataset statistics from arXiv HTML v2 via fetch tool; licence read from repo file; venue from NeurIPS poster page search result.

## Relevance to this project
- Do not use GenImage alone for conclusions: format/size bias; use format-matched variants.
- Mostly pre-2023 generators and low resolutions; weak proxy for modern unseen generators.
