# Recorded new E4 reference results

These are the existing completed E4 evaluation records, imported during PMGS packaging. No model was retrained to produce this table. Each scene was evaluated at iteration 30,000 on 200 original 800×800 test views with a black background.

PSNR is the historical mean of per-channel RGB PSNR, not joint-RGB PSNR. SSIM and VGG LPIPS v0.1 were computed on floating-point rendered RGB before saving PNG. Different PSNR conventions should not be compared as if they were identical.

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Final Gaussians |
|---|---:|---:|---:|---:|
| chair | 35.7769 | 0.986991 | 0.014310 | 293,581 |
| drums | 26.3373 | 0.954188 | 0.041979 | 489,661 |
| ficus | 35.1948 | 0.985856 | 0.014091 | 322,228 |
| hotdog | 38.0910 | 0.985424 | 0.025928 | 135,530 |
| lego | 36.7864 | 0.983918 | 0.016055 | 1,122,799 |
| materials | 30.4025 | 0.957973 | 0.041460 | 510,551 |
| mic | 37.3178 | 0.993341 | 0.007438 | 341,305 |
| ship | 31.8595 | 0.904771 | 0.113502 | 1,027,049 |
| Mean | 33.9708 | 0.969058 | 0.034345 | — |

Full recorded precision and initial/final Gaussian counts are in [reference_results.csv](reference_results.csv). The source records were `E4_retained/<scene>/attempt_01/evaluation.json` from the completed E4 experiment. The package does not include those checkpoints or rendered images.

The configuration is recorded in `configs/e4.json`; exact Qgain replay requires the original dataset and mesh signatures. These records do not establish a new benchmark against other methods.
