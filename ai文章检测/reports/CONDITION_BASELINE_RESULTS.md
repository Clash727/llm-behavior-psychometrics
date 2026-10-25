# Condition-prefix baseline results

This report summarizes the completed `condition_baseline_v2` CPU run in `artifacts/condition_baseline/formal_retry1`. It uses saved probabilities and metrics only; no model was fit for this report.

The IID row is the frozen dev set. Generator and WikiHow rows are development pressure comparisons whose source families/splits have already informed development; they are not blind confirmation or a final generalization estimate. Every comparison selected HGB on its fixed S partition. All six comparison sets use the same frozen 448-token singleton-scored probe and role manifests.

## Main results

| Evaluation | Comparison | Overall macro-F1 | Short macro-F1 | Combined | Human FPR | Machine recall | ROC AUC | n / short n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IID dev | A | 0.819 | 0.872 | 0.835 | 0.257 | 0.895 | 0.921 | 13,387 / 1,975 |
| IID dev | B | 0.887 | 0.923 | 0.898 | 0.162 | 0.936 | 0.967 | 13,387 / 1,975 |
| IID dev | C_neutral | 0.901 | 0.909 | 0.904 | 0.146 | 0.946 | 0.974 | 13,387 / 1,975 |
| IID dev | C | 0.911 | 0.910 | 0.910 | 0.135 | 0.953 | 0.978 | 13,387 / 1,975 |
| IID dev | D_global | 0.911 | 0.915 | 0.912 | 0.133 | 0.952 | 0.978 | 13,387 / 1,975 |
| IID dev | D | 0.913 | 0.911 | 0.912 | 0.131 | 0.953 | 0.979 | 13,387 / 1,975 |
| Generator OOD eval | A | 0.722 | 0.854 | 0.761 | 0.250 | 0.694 | 0.801 | 32,132 / 5,513 |
| Generator OOD eval | B | 0.768 | 0.864 | 0.797 | 0.224 | 0.759 | 0.853 | 32,132 / 5,513 |
| Generator OOD eval | C_neutral | 0.775 | 0.859 | 0.800 | 0.205 | 0.754 | 0.861 | 32,132 / 5,513 |
| Generator OOD eval | C | 0.777 | 0.863 | 0.803 | 0.178 | 0.732 | 0.863 | 32,132 / 5,513 |
| Generator OOD eval | D_global | 0.777 | 0.859 | 0.801 | 0.182 | 0.735 | 0.862 | 32,132 / 5,513 |
| Generator OOD eval | D | 0.777 | 0.861 | 0.802 | 0.179 | 0.734 | 0.865 | 32,132 / 5,513 |
| WikiHow OOD eval | A | 0.755 | 0.407 | 0.651 | 0.153 | 0.795 | 0.903 | 12,703 / 35 |
| WikiHow OOD eval | B | 0.737 | 0.407 | 0.638 | 0.030 | 0.722 | 0.960 | 12,703 / 35 |
| WikiHow OOD eval | C_neutral | 0.737 | 0.397 | 0.635 | 0.030 | 0.722 | 0.958 | 12,703 / 35 |
| WikiHow OOD eval | C | 0.760 | 0.407 | 0.654 | 0.044 | 0.758 | 0.956 | 12,703 / 35 |
| WikiHow OOD eval | D_global | 0.761 | 0.407 | 0.655 | 0.043 | 0.759 | 0.955 | 12,703 / 35 |
| WikiHow OOD eval | D | 0.766 | 0.397 | 0.655 | 0.043 | 0.765 | 0.955 | 12,703 / 35 |

Combined score is `0.7 × overall macro-F1 + 0.3 × short macro-F1`, with short texts defined as fewer than 100 whitespace tokens. FPR and recall use the fixed 0.5 threshold. WikiHow eval has only 35 short examples, so its short and combined estimates are noisy and should not drive conclusions.

### What the comparisons show

- C-neutral−B combined-score point differences are +0.006 on IID, +0.003 on generator eval, and −0.003 on WikiHow eval. The neutral response adds a small, non-uniform signal.
- C−C-neutral combined-score points are positive on all three evaluations (+0.007 IID, +0.003 generator, +0.020 WikiHow). The paired macro-F1 interval excludes zero for IID and WikiHow but includes zero for generator eval, so this run does not establish transferable human/machine-hypothesis benefit across unknown generators. WikiHow has only 35 short examples.
- C−B combined-score point estimates are positive on all three evaluations (+0.013, +0.006, +0.016). Much of the IID improvement already appears in B−A text-only background features.
- WikiHow is a counterexample to assuming text-only features always improve fixed-threshold classification: B−A lowers macro-F1 by 0.018 and combined score by 0.013, although AUC rises by 0.057 and FPR falls sharply; machine recall also falls. This is a ranking/threshold tradeoff on this pressure set.
- D−C combined-score points are +0.002 IID, −0.001 generator, and +0.001 WikiHow. D uses extra R-human labels, so D−C cannot isolate conditionalization. WikiHow D−C macro-F1 is positive in the paired interval, but short macro-F1 declines and the combined difference is nearly zero. D-global uses the same R-human rows as D; D−D-global is effectively zero across protocols, so conditional reference cells show no stable combined-score advantage.
- The script grouping is only a coarse Unicode script proxy, not language identification. No source/generator/language labels route predictions.

## Paired group-bootstrap differences

Intervals below are percentile 95% CIs from the 200 paired connected-component resamples already saved by the formal run (seed `20261020`). Each comparison pair uses identical resampled groups. Positive values favor the left side of the contrast. The formal run did not save bootstrap draws for short macro-F1 or combined score, so those are point differences only; the paired CIs below apply only to saved macro-F1 draws. See `plots/paired_bootstrap_differences.csv` for all point differences and available intervals.

| Evaluation | Contrast | Δ combined (point) | Δ short macro-F1 (point) | Δ macro-F1 (95% CI) |
|---|---|---:|---:|---:|
| IID dev | Cneutral−B | 0.006 | -0.014 | 0.014 (0.009, 0.019) |
| IID dev | C−Cneutral | 0.007 | 0.001 | 0.009 (0.006, 0.013) |
| IID dev | C−B | 0.013 | -0.013 | 0.024 (0.017, 0.029) |
| IID dev | D−C | 0.002 | 0.001 | 0.002 (-0.001, 0.005) |
| IID dev | D−Dglobal | -0.000 | -0.004 | 0.002 (-0.001, 0.005) |
| Generator OOD eval | Cneutral−B | 0.003 | -0.005 | 0.007 (0.004, 0.011) |
| Generator OOD eval | C−Cneutral | 0.003 | 0.005 | 0.002 (-0.001, 0.005) |
| Generator OOD eval | C−B | 0.006 | -0.001 | 0.009 (0.005, 0.014) |
| Generator OOD eval | D−C | -0.001 | -0.002 | 0.000 (-0.002, 0.003) |
| Generator OOD eval | D−Dglobal | 0.001 | 0.002 | 0.001 (-0.002, 0.003) |
| WikiHow OOD eval | Cneutral−B | -0.003 | -0.010 | -0.000 (-0.005, 0.004) |
| WikiHow OOD eval | C−Cneutral | 0.020 | 0.010 | 0.024 (0.020, 0.028) |
| WikiHow OOD eval | C−B | 0.016 | 0.000 | 0.023 (0.018, 0.029) |
| WikiHow OOD eval | D−C | 0.001 | -0.010 | 0.005 (0.003, 0.008) |
| WikiHow OOD eval | D−Dglobal | -0.000 | -0.010 | 0.004 (0.001, 0.008) |

These intervals quantify variation across the already-fixed held evaluation rows; they do not make the OOD pressure sets blind or remove the small-short-sample limitation. The exact 200 bootstrap draws and source probability rows remain in the formal run artifacts.

## Reference and supervision budget

R/F/S were assigned once by complete connected article/body-alias components. Reference rows were human-only within R (IID 4,895; generator 2,129; WikiHow 4,466). F trained the fixed LR/HGB candidates; S selected the classifier family; no dev/eval data fit a transform or selected a model. D and D-global both use R-human reference statistics; D adds bucketed length/script-proxy cells, with under-200 cells falling back to the global reference. D versus C has a larger supervision budget, while D versus D-global compares conditional to global reference at the same R-human budget.

## Decision and system reference

C is a useful auxiliary research signal, not a replacement for the current Qwen classifier. The existing system reference scores are IID 0.991928, generator 0.950560, and WikiHow 0.901386, compared with C combined scores 0.91046, 0.80292, and 0.65433. These are system-level references with different architectures/data budgets, not a single-factor comparison. Stop this branch at the current six comparisons; do not start Template B, pseudo-labeling, or cross-reference-family transfer automatically.

## Run cost

The singleton GPU extraction took 30,582.99 seconds (8.50 hours) for 98,756 documents at four conditions and a 448-token body budget; this includes I/O, averaging 0.30968 seconds/document, with peak allocated memory 8,100,402,176 bytes. CPU `formal_retry1` took 772.527 seconds (12 minutes 52.5 seconds) under the two-thread limit. These are end-to-end run costs, not isolated model-forward benchmarks.

## Figures and plotting data

- `artifacts/condition_baseline/formal_retry1/plots/scores.png` and `.svg`: overall, short, and combined scores.
- `.../plots/operating_points.png` and `.svg`: FPR/recall operating points.
- `.../plots/roc.png` and `.svg`: ROC curves.
- `.../plots/calibration.png` and `.svg`: probability calibration.
- `.../plots/paired_macro_differences.png` and `.svg`: paired bootstrap CIs for macro-F1 contrasts.
- CSV files alongside the figures retain the metric table, ROC points, calibration bins, and paired-difference CIs.

The model/probability/report artifacts were left unchanged. This summary does not initiate template B, additional tuning, or any new GPU extraction.
