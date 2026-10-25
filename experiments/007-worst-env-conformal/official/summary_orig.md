# Summary OFFICIAL H2-v2, arm orig

## PRIMARY: platform-only groups, min-group 100
| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |
|---|---|---|---|---|---|---|---|---|
| B0 | 1.34% / 1.28% / 0.78% | 2.23% (WA) / 2.51% (WA) / 1.96% (WA) | 2.51% / 1.96% | 38.63% -> 28.17% | 29.12% -> 20.83% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| B0_twinfree | 2.29% / 1.56% / 1.17% | 3.07% (jpeg75_resize05) / 2.51% (WA) / 1.96% (WA) | 2.51% / 1.96% | 31.38% -> 25.98% | 22.63% -> 16.13% | 0.00 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| bfree | 3.80% / 0.95% / 0.61% | 4.47% (WA) / 3.07% (WA) / 2.79% (WA) | 3.07% / 2.79% | 69.36% -> 62.29% | 17.06% -> 12.72% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| clipdet | 11.12% / 7.43% / 3.63% | 38.27% (jpeg75_resize05) / 27.37% (jpeg75_resize05) / 12.57% (jpeg75_resize05) | 27.37% / 12.57% | 17.12% -> 11.24% | 21.85% -> 14.41% | 0.00 | SUPPORTED (3 CI<0, 1 diff>=0) | True |
| communityforensics | 2.29% / 1.40% / 0.89% | 3.91% (screenshot_sim) / 2.51% (screenshot_sim) / 1.68% (native) | 2.51% / 1.68% | 89.39% -> 77.26% | 32.62% -> 18.31% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| rine | 0.50% / 0.67% / 0.50% | 1.96% (native) / 2.51% (native) / 2.23% (native) | 2.51% / 2.23% | 60.66% -> 58.26% | 37.79% -> 30.56% | 0.00 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |

## SENSITIVITY: grouping platform_brand, min-group 50
| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |
|---|---|---|---|---|---|---|---|---|
| B0 | 1.34% / 1.28% / 0.34% | 2.23% (WA) / 2.51% (WA) / 1.12% (WA) | 6.09% / 2.61% | 38.63% -> 16.94% | 29.12% -> 9.23% | 0.57 | INCONCLUSIVE (0 CI<0, 2 diff>=0) | True |
| B0_twinfree | 2.29% / 1.56% / 0.22% | 3.07% (jpeg75_resize05) / 2.51% (WA) / 0.56% (WA) | 6.96% / 1.15% | 31.38% -> 17.26% | 22.63% -> 7.76% | 0.57 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| bfree | 3.80% / 0.95% / 0.34% | 4.47% (WA) / 3.07% (WA) / 1.68% (WA) | 10.53% / 7.02% | 69.36% -> 34.09% | 17.06% -> 5.08% | 0.57 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| clipdet | 11.12% / 7.43% / 1.06% | 38.27% (jpeg75_resize05) / 27.37% (jpeg75_resize05) / 3.35% (jpeg75_resize05) | 40.35% / 6.09% | 17.12% -> 5.61% | 21.85% -> 6.90% | 0.57 | INCONCLUSIVE (1 CI<0, 2 diff>=0) | True |
| communityforensics | 2.29% / 1.40% / 0.50% | 3.91% (screenshot_sim) / 2.51% (screenshot_sim) / 1.68% (native) | 12.28% / 10.53% | 89.39% -> 72.10% | 32.62% -> 13.48% | 0.57 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| rine | 0.50% / 0.67% / 0.17% | 1.96% (native) / 2.51% (native) / 0.84% (native) | 5.75% / 1.75% | 60.66% -> 53.18% | 37.79% -> 19.58% | 0.57 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |

## SENSITIVITY: grouping platform, min-group 50
| detector | LODO pooled FPR M1 / M2(pooled@1%) / wg@1% | worst-platform FPR (LOPO view) M1 / M2 / wg@1% | worst-group FPR M2 / wg | TPR clean M2 -> wg@1% | TPR jpeg M2 -> wg | clipped wg share @1% | verdict | problem real |
|---|---|---|---|---|---|---|---|---|
| B0 | 1.34% / 1.28% / 0.78% | 2.23% (WA) / 2.51% (WA) / 1.96% (WA) | 2.51% / 1.96% | 38.63% -> 28.17% | 29.12% -> 20.83% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| B0_twinfree | 2.29% / 1.56% / 1.17% | 3.07% (jpeg75_resize05) / 2.51% (WA) / 1.96% (WA) | 2.51% / 1.96% | 31.38% -> 25.98% | 22.63% -> 16.13% | 0.00 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |
| bfree | 3.80% / 0.95% / 0.61% | 4.47% (WA) / 3.07% (WA) / 2.79% (WA) | 3.07% / 2.79% | 69.36% -> 62.29% | 17.06% -> 12.72% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| clipdet | 11.12% / 7.43% / 3.63% | 38.27% (jpeg75_resize05) / 27.37% (jpeg75_resize05) / 12.57% (jpeg75_resize05) | 27.37% / 12.57% | 17.12% -> 11.24% | 21.85% -> 14.41% | 0.00 | SUPPORTED (3 CI<0, 1 diff>=0) | True |
| communityforensics | 2.29% / 1.40% / 0.89% | 3.91% (screenshot_sim) / 2.51% (screenshot_sim) / 1.68% (native) | 2.51% / 1.68% | 89.39% -> 77.26% | 32.62% -> 18.31% | 0.00 | NOT_SUPPORTED (0 CI<0, 3 diff>=0) | True |
| rine | 0.50% / 0.67% / 0.50% | 1.96% (native) / 2.51% (native) / 2.23% (native) | 2.51% / 2.23% | 60.66% -> 58.26% | 37.79% -> 30.56% | 0.00 | NOT_SUPPORTED (0 CI<0, 4 diff>=0) | True |

## Verdict robustness (Amendment 13): primary verdict with Firefly out / each generator dropped
| detector | full | dalle2 | dalle3 | firefly | flux | midjourney-v5 | sd3_large | stable-diffusion-2 | stable-diffusion-xl |
|---|---|---|---|---|---|---|---|---|---|
| B0 | NS | NS | NS | NS | NS | NS | NS | NS | NS |
| B0_twinfree | NS | NS | NS | NS | NS | NS | NS | NS | NS |
| bfree | NS | NS | NS | NS | NS | NS | NS | NS | NS |
| clipdet | SUP | SUP | SUP | SUP | SUP | SUP | SUP | SUP | SUP |
| communityforensics | NS | NS | NS | NS | NS | NS | NS | NS | NS |
| rine | NS | NS | NS | NS | NS | NS | NS | NS | NS |

Sensitivity (brand-50) with Firefly out: B0: INCONCLUSIVE; B0_twinfree: NOT_SUPPORTED; bfree: NOT_SUPPORTED; clipdet: INCONCLUSIVE; communityforensics: NOT_SUPPORTED; rine: NOT_SUPPORTED
Primary groups: FBH n=358; WA n=358; native n=358
Sensitivity groups: FBH/Apple n=99; FBH/Huawei n=57; FBH/Samsung n=87; FBH/other n=115; WA/Apple n=99; WA/Huawei n=57; WA/Samsung n=87; WA/other n=115; native/Apple n=99; native/Huawei n=57; native/Samsung n=87; native/other n=115
