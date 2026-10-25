#!/usr/bin/env python3
"""Summarize released condition-baseline artifacts without fitting models."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from b_text_detector.ood import connected_components, load_development_pool  # noqa: E402

OUT = ROOT / "artifacts/condition_baseline/formal_retry1"
PLOTS = OUT / "plots"
CONFIG = ROOT / "configs/condition_baseline.json"
TRAIN = ROOT / "artifacts/audited_train/splits/train.jsonl"
DEV = ROOT / "artifacts/audited_train/splits/dev.jsonl"
MANIFESTS = ROOT / "artifacts/condition_baseline/manifests"
OOD_ROOT = ROOT / "artifacts/experiments/ood"
ORDER = ["A", "B", "C_neutral", "C", "D_global", "D"]
LABELS = {"A": "A · raw", "B": "B · +text", "C_neutral": "C-neutral · +neutral Δ",
          "C": "C · +H/M Δ", "D_global": "D-global · +global ref", "D": "D · +conditional ref"}
METRICS = ("macro_f1", "short_macro_f1", "combined_score", "human_fpr", "machine_recall",
           "roc_auc", "brier", "log_loss")
CONTRASTS = [("B", "A"), ("C_neutral", "B"), ("C", "B"), ("C", "C_neutral"),
             ("D_global", "C"), ("D", "C"), ("D", "D_global")]
CONTRAST_LABELS = {("B", "A"): "B−A", ("C_neutral", "B"): "Cneutral−B",
                   ("C", "B"): "C−B", ("C", "C_neutral"): "C−Cneutral",
                   ("D_global", "C"): "Dglobal−C", ("D", "C"): "D−C",
                   ("D", "D_global"): "D−Dglobal"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics_for(y: np.ndarray, p: np.ndarray, short: np.ndarray) -> dict[str, float | int]:
    pred = (p >= 0.5).astype(np.int8)
    macro = float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
    if short.any():
        short_macro = float(f1_score(y[short], pred[short], labels=[0, 1], average="macro", zero_division=0))
    else:
        short_macro = macro
    return {"macro_f1": macro, "short_macro_f1": short_macro,
            "combined_score": .7 * macro + .3 * short_macro,
            "human_fpr": float(np.mean(pred[y == 0])) if np.any(y == 0) else float("nan"),
            "machine_recall": float(np.mean(pred[y == 1])) if np.any(y == 1) else float("nan")}


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float("nan"), float("nan"), float("nan")
    return float(arr.mean()), float(np.quantile(arr, .025)), float(np.quantile(arr, .975))


def calibration(y: np.ndarray, probs: dict[str, np.ndarray], bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for name, p in probs.items():
        for b in range(bins):
            mask = ((p >= edges[b]) & (p < edges[b + 1])) if b < bins - 1 else ((p >= edges[b]) & (p <= edges[b + 1]))
            if mask.any():
                result.append({"comparison": name, "bin": b, "n": int(mask.sum()),
                               "mean_probability": float(p[mask].mean()),
                               "fraction_machine": float(y[mask].mean())})
    return result


def load_cohorts(report: dict) -> dict[str, dict]:
    cohorts = {}
    pool = load_development_pool(TRAIN, DEV)
    iid_dev = [row for row in pool if row["_original_split"] == "dev"]
    iid_predictions = read_jsonl(OUT / "dev_predictions.jsonl")
    _, iid_groups = connected_components(iid_dev)
    if len(iid_dev) != len(iid_predictions):
        raise ValueError("IID dev predictions do not match frozen dev rows")
    for i, (source, pred) in enumerate(zip(iid_dev, iid_predictions)):
        if int(source["label"]) != int(pred["label"]) or int(pred["original_index"]) != i:
            raise ValueError(f"IID dev row identity mismatch at {i}")
    cohorts["IID dev"] = {
        "key": "iid", "kind": "IID frozen dev", "n": len(iid_dev),
        "y": np.asarray([int(row["label"]) for row in iid_dev], dtype=np.int8),
        "short": np.asarray([len(str(row["text"]).split()) < 100 for row in iid_dev]),
        "groups": np.asarray(iid_groups),
        "probabilities": {name: np.asarray([float(row[f"{name}_probability"]) for row in iid_predictions])
                          for name in ORDER},
        "metrics": {name: report["selection_and_dev"][name]["dev_metrics"] for name in ORDER},
        "selection": {name: report["selection_and_dev"][name] for name in ORDER},
        "bootstrap": report["bootstrap_draws"],
    }

    for family, display in (("generator_davinci", "Generator OOD eval"), ("domain_wikihow", "WikiHow OOD eval")):
        pred_rows = read_jsonl(OUT / family / "eval_predictions.jsonl")
        source_rows = read_jsonl(OOD_ROOT / family / "eval.jsonl")
        lineage = read_jsonl(MANIFESTS / f"{family}.eval.roles.jsonl")
        if not (len(pred_rows) == len(source_rows) == len(lineage)):
            raise ValueError(f"{family}: eval prediction/source/lineage row counts differ")
        for i, (pred, row, identity) in enumerate(zip(pred_rows, source_rows, lineage)):
            if (pred["row_key"] != identity["row_key"] or int(pred["label"]) != int(row["label"])
                    or int(pred["label"]) != int(identity["label"])):
                raise ValueError(f"{family}: held-eval row alignment mismatch at {i}")
        ood = report["ood_protocol_results"][family]
        cohorts[display] = {
            "key": family, "kind": "development pressure set · held eval",
            "n": len(source_rows), "y": np.asarray([int(row["label"]) for row in source_rows], dtype=np.int8),
            "short": np.asarray([len(str(row["text"]).split()) < 100 for row in source_rows]),
            "groups": np.asarray([int(row["group_id"]) for row in lineage], dtype=np.int64),
            "probabilities": {name: np.asarray([float(row[f"{name}_probability"]) for row in pred_rows])
                              for name in ORDER},
            "metrics": {name: ood["selection_and_eval"][name]["dev_metrics"] for name in ORDER},
            "selection": {name: ood["selection_and_eval"][name] for name in ORDER},
            "bootstrap": ood["eval_bootstrap_draws"],
        }
    return cohorts


def collect_tables(report: dict, cohorts: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    metric_rows, roc_rows, calib_rows, delta_rows = [], [], [], []
    for dataset, cohort in cohorts.items():
        for name in ORDER:
            m = cohort["metrics"][name]
            selected = cohort["selection"][name]
            metric_rows.append({"dataset": dataset, "dataset_kind": cohort["kind"], "comparison": name,
                "selected_classifier": selected["selected_classifier"], "n": m["n"], "short_n": m["short_n"],
                "selection_combined_score": selected["selected_s_combined_score"],
                **{metric: m.get(metric) for metric in METRICS}})
            for point in m["roc"]:
                roc_rows.append({"dataset": dataset, "comparison": name, **point})
        calibration_rows = calibration(cohort["y"], cohort["probabilities"])
        for row in calibration_rows:
            calib_rows.append({"dataset": dataset, **row})

        for left, right in CONTRASTS:
            paired = {metric: [] for metric in METRICS}
            saved_metric = ("macro_f1", "roc_auc", "brier", "human_fpr", "machine_recall")
            left_saved, right_saved = cohort["bootstrap"][left], cohort["bootstrap"][right]
            for metric in saved_metric:
                n = min(len(left_saved), len(right_saved))
                paired[metric] = [float(left_saved[i][metric]) - float(right_saved[i][metric]) for i in range(n)]
            point_metrics = cohort["metrics"][left]
            other_metrics = cohort["metrics"][right]
            estimates = {}
            for metric in METRICS:
                estimates[metric] = float(point_metrics[metric]) - float(other_metrics[metric])
                mean, low, high = percentile_ci(paired[metric]) if paired[metric] else (None, None, None)
                delta_rows.append({"dataset": dataset, "contrast": CONTRAST_LABELS[(left, right)],
                    "left": left, "right": right, "metric": metric, "point_difference": estimates[metric],
                    "bootstrap_mean_difference": mean, "ci_2_5": low, "ci_97_5": high,
                    "replicates": len(paired[metric]),
                    "bootstrap_unit": "connected_component" if paired[metric] else "not_saved_in_formal_run",
                    "seed": 20261020})
    return metric_rows, roc_rows, calib_rows, delta_rows


def make_plots(metric_rows: list[dict], roc_rows: list[dict], calib_rows: list[dict], delta_rows: list[dict]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    datasets = ["IID dev", "Generator OOD eval", "WikiHow OOD eval"]
    colors = dict(zip(datasets, ("#2463a6", "#d27a22", "#3a8b64")))
    comparisons = ORDER
    x = np.arange(len(comparisons))
    aliases = {"macro_f1": "Overall macro-F1", "short_macro_f1": "Short macro-F1",
               "combined_score": "Combined score"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, aliases):
        for dataset in datasets:
            values = [next(r[metric] for r in metric_rows if r["dataset"] == dataset and r["comparison"] == name)
                      for name in comparisons]
            ax.plot(x, values, marker="o", linewidth=2, label=dataset, color=colors[dataset])
        ax.set_title(aliases[metric])
        ax.set_xticks(x, ["A", "B", "Cₙ", "C", "Dg", "D"], rotation=0)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylim(0.35, 1.0)
    axes[0].set_ylabel("Score at threshold 0.5")
    axes[-1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Condition-baseline comparisons · selected HGB\nShort sample n: IID 1,975 · Generator 5,513 · WikiHow 35", fontsize=14)
    fig.savefig(PLOTS / "scores.png", dpi=180)
    fig.savefig(PLOTS / "scores.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True)
    palette = dict(zip(comparisons, ("#555555", "#8a6d3b", "#327a54", "#2463a6", "#8d4f94", "#d04a35")))
    label_offsets = ((4, 5), (4, -9), (4, 12), (4, -16), (4, 19), (4, -23))
    for ax, dataset in zip(axes, datasets):
        for comparison_ix, name in enumerate(comparisons):
            row = next(r for r in metric_rows if r["dataset"] == dataset and r["comparison"] == name)
            ax.scatter(row["human_fpr"], row["machine_recall"], label=name,
                       color=palette[name], s=48, marker="o")
            ax.annotate(name, (row["human_fpr"], row["machine_recall"]), xytext=label_offsets[comparison_ix],
                        textcoords="offset points", fontsize=7)
        ax.set_title(dataset)
        ax.grid(alpha=.2)
        ax.set_xlabel("Human false positive rate")
        ax.set_xlim(0, max(.28, ax.get_xlim()[1]))
        ax.set_ylim(.65, 1.0)
    axes[0].set_ylabel("Machine recall")
    fig.suptitle("Operating points · threshold 0.5", fontsize=14)
    fig.savefig(PLOTS / "operating_points.png", dpi=180)
    fig.savefig(PLOTS / "operating_points.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        for name in comparisons:
            rows = sorted((r for r in roc_rows if r["dataset"] == dataset and r["comparison"] == name),
                          key=lambda row: row["fpr"])
            ax.plot([r["fpr"] for r in rows], [r["tpr"] for r in rows], linewidth=1.3, label=name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        ax.set_title(dataset)
        ax.grid(alpha=.2)
        ax.set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("ROC curves · frozen IID dev and development pressure evals", fontsize=14)
    fig.savefig(PLOTS / "roc.png", dpi=180)
    fig.savefig(PLOTS / "roc.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        for name in comparisons:
            rows = sorted((r for r in calib_rows if r["dataset"] == dataset and r["comparison"] == name),
                          key=lambda row: row["bin"])
            ax.plot([r["mean_probability"] for r in rows], [r["fraction_machine"] for r in rows],
                    marker="o", markersize=3, linewidth=1.2, label=name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        ax.set_title(dataset)
        ax.grid(alpha=.2)
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed machine fraction")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Calibration · equal-width probability bins", fontsize=14)
    fig.savefig(PLOTS / "calibration.png", dpi=180)
    fig.savefig(PLOTS / "calibration.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        rows = [r for r in delta_rows if r["dataset"] == dataset and r["metric"] == "macro_f1"]
        ypos = np.arange(len(rows))
        values = np.asarray([r["point_difference"] for r in rows])
        low = np.asarray([r["ci_2_5"] for r in rows])
        high = np.asarray([r["ci_97_5"] for r in rows])
        ax.errorbar(values, ypos, xerr=np.vstack([values - low, high - values]), fmt="o", capsize=2,
                    color=colors[dataset])
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_yticks(ypos, [r["contrast"] for r in rows])
        ax.set_title(dataset)
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Δ macro-F1 (95% paired group CI)")
    fig.suptitle("Paired connected-component bootstrap differences", fontsize=14)
    fig.savefig(PLOTS / "paired_macro_differences.png", dpi=180)
    fig.savefig(PLOTS / "paired_macro_differences.svg")
    plt.close(fig)


def write_report(report: dict, cohorts: dict[str, dict], metric_rows: list[dict], delta_rows: list[dict]) -> None:
    def f(value, digits=3):
        return "—" if value is None else f"{float(value):.{digits}f}"

    lines = ["# Condition-prefix baseline results", "",
        "This report summarizes the completed `condition_baseline_v2` CPU run in `artifacts/condition_baseline/formal_retry1`. It uses saved probabilities and metrics only; no model was fit for this report.", "",
        "The IID row is the frozen dev set. Generator and WikiHow rows are development pressure comparisons whose source families/splits have already informed development; they are not blind confirmation or a final generalization estimate. Every comparison selected HGB on its fixed S partition. All six comparison sets use the same frozen 448-token singleton-scored probe and role manifests.", "",
        "## Main results", "",
        "| Evaluation | Comparison | Overall macro-F1 | Short macro-F1 | Combined | Human FPR | Machine recall | ROC AUC | n / short n |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in cohorts:
        for name in ORDER:
            row = next(r for r in metric_rows if r["dataset"] == dataset and r["comparison"] == name)
            lines.append(f"| {dataset} | {name} | {f(row['macro_f1'])} | {f(row['short_macro_f1'])} | {f(row['combined_score'])} | {f(row['human_fpr'])} | {f(row['machine_recall'])} | {f(row['roc_auc'])} | {row['n']:,} / {row['short_n']:,} |")

    lines += ["", "Combined score is `0.7 × overall macro-F1 + 0.3 × short macro-F1`, with short texts defined as fewer than 100 whitespace tokens. FPR and recall use the fixed 0.5 threshold. WikiHow eval has only 35 short examples, so its short and combined estimates are noisy and should not drive conclusions.", "",
        "### What the comparisons show", "",
        "- C-neutral−B combined-score point differences are +0.006 on IID, +0.003 on generator eval, and −0.003 on WikiHow eval. The neutral response adds a small, non-uniform signal.",
        "- C−C-neutral combined-score points are positive on all three evaluations (+0.007 IID, +0.003 generator, +0.020 WikiHow). The paired macro-F1 interval excludes zero for IID and WikiHow but includes zero for generator eval, so this run does not establish transferable human/machine-hypothesis benefit across unknown generators. WikiHow has only 35 short examples.",
        "- C−B combined-score point estimates are positive on all three evaluations (+0.013, +0.006, +0.016). Much of the IID improvement already appears in B−A text-only background features.",
        "- WikiHow is a counterexample to assuming text-only features always improve fixed-threshold classification: B−A lowers macro-F1 by 0.018 and combined score by 0.013, although AUC rises by 0.057 and FPR falls sharply; machine recall also falls. This is a ranking/threshold tradeoff on this pressure set.",
        "- D−C combined-score points are +0.002 IID, −0.001 generator, and +0.001 WikiHow. D uses extra R-human labels, so D−C cannot isolate conditionalization. WikiHow D−C macro-F1 is positive in the paired interval, but short macro-F1 declines and the combined difference is nearly zero. D-global uses the same R-human rows as D; D−D-global is effectively zero across protocols, so conditional reference cells show no stable combined-score advantage.",
        "- The script grouping is only a coarse Unicode script proxy, not language identification. No source/generator/language labels route predictions.", "",
        "## Paired group-bootstrap differences", "",
        "Intervals below are percentile 95% CIs from the 200 paired connected-component resamples already saved by the formal run (seed `20261020`). Each comparison pair uses identical resampled groups. Positive values favor the left side of the contrast. The formal run did not save bootstrap draws for short macro-F1 or combined score, so those are point differences only; the paired CIs below apply only to saved macro-F1 draws. See `plots/paired_bootstrap_differences.csv` for all point differences and available intervals.", "",
        "| Evaluation | Contrast | Δ combined (point) | Δ short macro-F1 (point) | Δ macro-F1 (95% CI) |", "|---|---|---:|---:|---:|"]
    for dataset in cohorts:
        for contrast in ("Cneutral−B", "C−Cneutral", "C−B", "D−C", "D−Dglobal"):
            vals = {}
            for metric in ("combined_score", "macro_f1", "short_macro_f1"):
                r = next(x for x in delta_rows if x["dataset"] == dataset and x["contrast"] == contrast and x["metric"] == metric)
                vals[metric] = (f"{f(r['point_difference'])} ({f(r['ci_2_5'])}, {f(r['ci_97_5'])})"
                                if metric == "macro_f1" else f(r['point_difference']))
            lines.append(f"| {dataset} | {contrast} | {vals['combined_score']} | {vals['short_macro_f1']} | {vals['macro_f1']} |")
    lines += ["", "These intervals quantify variation across the already-fixed held evaluation rows; they do not make the OOD pressure sets blind or remove the small-short-sample limitation. The exact 200 bootstrap draws and source probability rows remain in the formal run artifacts.", "",
        "## Reference and supervision budget", "",
        "R/F/S were assigned once by complete connected article/body-alias components. Reference rows were human-only within R (IID 4,895; generator 2,129; WikiHow 4,466). F trained the fixed LR/HGB candidates; S selected the classifier family; no dev/eval data fit a transform or selected a model. D and D-global both use R-human reference statistics; D adds bucketed length/script-proxy cells, with under-200 cells falling back to the global reference. D versus C has a larger supervision budget, while D versus D-global compares conditional to global reference at the same R-human budget.", "",
        "## Decision and system reference", "",
        "C is a useful auxiliary research signal, not a replacement for the current Qwen classifier. The existing system reference scores are IID 0.991928, generator 0.950560, and WikiHow 0.901386, compared with C combined scores 0.91046, 0.80292, and 0.65433. These are system-level references with different architectures/data budgets, not a single-factor comparison. Stop this branch at the current six comparisons; do not start Template B, pseudo-labeling, or cross-reference-family transfer automatically.", "",
        "## Run cost", "",
        "The singleton GPU extraction took 30,582.99 seconds (8.50 hours) for 98,756 documents at four conditions and a 448-token body budget; this includes I/O, averaging 0.30968 seconds/document, with peak allocated memory 8,100,402,176 bytes. CPU `formal_retry1` took 772.527 seconds (12 minutes 52.5 seconds) under the two-thread limit. These are end-to-end run costs, not isolated model-forward benchmarks.", "",
        "## Figures and plotting data", "",
        "- `artifacts/condition_baseline/formal_retry1/plots/scores.png` and `.svg`: overall, short, and combined scores.",
        "- `.../plots/operating_points.png` and `.svg`: FPR/recall operating points.",
        "- `.../plots/roc.png` and `.svg`: ROC curves.",
        "- `.../plots/calibration.png` and `.svg`: probability calibration.",
        "- `.../plots/paired_macro_differences.png` and `.svg`: paired bootstrap CIs for macro-F1 contrasts.",
        "- CSV files alongside the figures retain the metric table, ROC points, calibration bins, and paired-difference CIs.", "",
        "The model/probability/report artifacts were left unchanged. This summary does not initiate template B, additional tuning, or any new GPU extraction.", ""]
    (ROOT / "reports/CONDITION_BASELINE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    status_path = OUT / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "complete" or status.get("failed") is not False:
        raise SystemExit("formal_retry1 is not complete; refusing to summarize")
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    if report.get("run_kind") != "formal":
        raise SystemExit("report is not a formal run")
    cohorts = load_cohorts(report)
    metric_rows, roc_rows, calib_rows, delta_rows = collect_tables(report, cohorts)
    PLOTS.mkdir(parents=True, exist_ok=True)
    save_csv(PLOTS / "metrics_by_dataset.csv", metric_rows,
        ["dataset", "dataset_kind", "comparison", "selected_classifier", "n", "short_n",
         "selection_combined_score", *METRICS])
    save_csv(PLOTS / "roc_points.csv", roc_rows, ["dataset", "comparison", "fpr", "tpr", "threshold"])
    save_csv(PLOTS / "calibration_bins.csv", calib_rows,
        ["dataset", "comparison", "bin", "n", "mean_probability", "fraction_machine"])
    save_csv(PLOTS / "paired_bootstrap_differences.csv", delta_rows,
        ["dataset", "contrast", "left", "right", "metric", "point_difference",
         "bootstrap_mean_difference", "ci_2_5", "ci_97_5", "replicates", "bootstrap_unit", "seed"])
    make_plots(metric_rows, roc_rows, calib_rows, delta_rows)
    write_report(report, cohorts, metric_rows, delta_rows)
    for dataset in cohorts:
        print(dataset, "n", cohorts[dataset]["n"], "short_n", int(cohorts[dataset]["short"].sum()))
        for name in ORDER:
            row = next(r for r in metric_rows if r["dataset"] == dataset and r["comparison"] == name)
            print(name, "selected", row["selected_classifier"],
                  "macro", round(row["macro_f1"], 4), "short", round(row["short_macro_f1"], 4),
                  "combined", round(row["combined_score"], 4), "fpr", round(row["human_fpr"], 4),
                  "recall", round(row["machine_recall"], 4))


if __name__ == "__main__":
    main()
