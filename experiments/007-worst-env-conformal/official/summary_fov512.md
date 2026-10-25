# Summary OFFICIAL H2-v2, arm fov512

## PRIMARY: platform-only groups, min-group 100
| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |
|---|---|---|---|---|---|---|---|---|
| B0 | 3.85% / 1.68% / 1.56% | 4.19% (native) / 1.68% (native) / 1.68% (native) | 1.68% / 1.68% | 22.21% -> 18.26% | 17.02% -> 14.30% | 0.00 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| B0_twinfree | 3.69% / 1.79% / 1.79% | 4.19% (WA) / 1.96% (native) / 1.96% (native) | 1.96% / 1.96% | 22.08% -> 18.02% | 16.79% -> 14.98% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| bfree | 6.59% / 1.68% / 0.95% | 12.01% (jpeg75_resize05) / 3.63% (jpeg75_resize05) / 1.68% (jpeg75_resize05) | 3.63% / 1.68% | 52.73% -> 46.63% | 16.73% -> 10.19% | 0.00 | INCONCLUSIVE (0 CI<0, 2 diff>=0) | True |
| clipdet | 0.06% / 2.63% / 1.79% | 0.28% (screenshot_sim) / 6.15% (jpeg75_resize05) / 4.19% (jpeg75_resize05) | 6.15% / 4.19% | 21.84% -> 15.90% | 18.71% -> 12.30% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| communityforensics | 5.87% / 1.51% / 1.28% | 8.66% (screenshot_sim) / 2.51% (screenshot_sim) / 1.96% (screenshot_sim) | 2.51% / 1.96% | 82.74% -> 82.14% | 16.37% -> 15.73% | 0.00 | INCONCLUSIVE (0 CI<0, 2 diff>=0) | True |
| rine | 10.56% / 3.41% / 1.34% | 29.61% (jpeg75_resize05) / 6.70% (jpeg75_resize05) / 5.03% (native) | 6.70% / 5.03% | 34.24% -> 24.18% | 26.29% -> 14.65% | 0.00 | NOT_SUPPORTED (1 CI<0, 3 diff>=0) | True |

## SENSITIVITY: grouping platform_brand, min-group 50
| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |
|---|---|---|---|---|---|---|---|---|
| B0 | 3.85% / 1.68% / 0.89% | 4.19% (native) / 1.68% (native) / 1.12% (native) | 5.22% / 3.48% | 22.21% -> 14.52% | 17.02% -> 11.39% | 0.57 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| B0_twinfree | 3.69% / 1.79% / 0.84% | 4.19% (WA) / 1.96% (native) / 1.12% (native) | 6.09% / 3.48% | 22.08% -> 12.63% | 16.79% -> 11.34% | 0.57 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| bfree | 6.59% / 1.68% / 0.22% | 12.01% (jpeg75_resize05) / 3.63% (jpeg75_resize05) / 0.56% (jpeg75_resize05) | 10.53% / 1.75% | 52.73% -> 39.16% | 16.73% -> 4.97% | 0.57 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| clipdet | 0.06% / 2.63% / 0.61% | 0.28% (screenshot_sim) / 6.15% (jpeg75_resize05) / 1.40% (jpeg75_resize05) | 19.30% / 5.26% | 21.84% -> 8.43% | 18.71% -> 4.92% | 0.57 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| communityforensics | 5.87% / 1.51% / 0.17% | 8.66% (screenshot_sim) / 2.51% (screenshot_sim) / 0.28% (native) | 7.02% / 0.87% | 82.74% -> 63.00% | 16.37% -> 7.03% | 0.57 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| rine | 10.56% / 3.41% / 0.84% | 29.61% (jpeg75_resize05) / 6.70% (jpeg75_resize05) / 3.63% (native) | 14.04% / 14.04% | 34.24% -> 18.39% | 26.29% -> 10.64% | 0.57 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |

## SENSITIVITY: grouping platform, min-group 50
| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |
|---|---|---|---|---|---|---|---|---|
| B0 | 3.85% / 1.68% / 1.56% | 4.19% (native) / 1.68% (native) / 1.68% (native) | 1.68% / 1.68% | 22.21% -> 18.26% | 17.02% -> 14.30% | 0.00 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| B0_twinfree | 3.69% / 1.79% / 1.79% | 4.19% (WA) / 1.96% (native) / 1.96% (native) | 1.96% / 1.96% | 22.08% -> 18.02% | 16.79% -> 14.98% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| bfree | 6.59% / 1.68% / 0.95% | 12.01% (jpeg75_resize05) / 3.63% (jpeg75_resize05) / 1.68% (jpeg75_resize05) | 3.63% / 1.68% | 52.73% -> 46.63% | 16.73% -> 10.19% | 0.00 | INCONCLUSIVE (0 CI<0, 2 diff>=0) | True |
| clipdet | 0.06% / 2.63% / 1.79% | 0.28% (screenshot_sim) / 6.15% (jpeg75_resize05) / 4.19% (jpeg75_resize05) | 6.15% / 4.19% | 21.84% -> 15.90% | 18.71% -> 12.30% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| communityforensics | 5.87% / 1.51% / 1.28% | 8.66% (screenshot_sim) / 2.51% (screenshot_sim) / 1.96% (screenshot_sim) | 2.51% / 1.96% | 82.74% -> 82.14% | 16.37% -> 15.73% | 0.00 | INCONCLUSIVE (0 CI<0, 2 diff>=0) | True |
| rine | 10.56% / 3.41% / 1.34% | 29.61% (jpeg75_resize05) / 6.70% (jpeg75_resize05) / 5.03% (native) | 6.70% / 5.03% | 34.24% -> 24.18% | 26.29% -> 14.65% | 0.00 | NOT_SUPPORTED (1 CI<0, 3 diff>=0) | True |

## Verdict robustness (Amendment 13): primary verdict with Firefly out / each generator dropped
| detector | full | dalle2 | dalle3 | firefly | flux | midjourney-v5 | sd3_large | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|---|---|---|
| B0 | NS | NS | NS | NS | NS | NS | NS | NS | NS |
| B0_twinfree | NS | NS | NS | NS | NS | NS | NS | NS | NS |
| bfree | INC | INC | INC | INC | INC | INC | INC | INC | INC |
| clipdet | NS | NS | INC | INC | NS | NS | NS | INC | NS |
| communityforensics | INC | INC | INC | INC | INC | INC | INC | INC | INC |
| rine | NS | NS | NS | NS | NS | NS | NS | NS | NS |

Sensitivity (brand-50) with Firefly out: B0: NOT_SUPPORTED; B0_twinfree: NOT_SUPPORTED; bfree: NOT_SUPPORTED; clipdet: NOT_SUPPORTED; communityforensics: NOT_SUPPORTED; rine: NOT_SUPPORTED
Primary groups: FBH n=358; WA n=358; native n=358
Sensitivity groups: FBH/Apple n=99; FBH/Huawei n=57; FBH/Samsung n=87; FBH/other n=115; WA/Apple n=99; WA/Huawei n=57; WA/Samsung n=87; WA/other n=115; native/Apple n=99; native/Huawei n=57; native/Samsung n=87; native/other n=115
