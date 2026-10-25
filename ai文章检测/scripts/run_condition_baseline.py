#!/usr/bin/env python3
"""Prepare role manifests or run the released condition-prefix CPU study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

# Keep BLAS/OpenMP pools bounded for the whole CPU pipeline, including OOD
# prediction and bootstrap. These must be set before importing NumPy/sklearn.
import os
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "2"

import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from b_text_detector.condition_baseline import (  # noqa: E402
    apply_human_reference,
    atomic_jsonl,
    build_feature_matrices,
    build_feature_matrices_stream,
    calibration_rows,
    fit_bounded_comparisons,
    fit_human_reference,
    make_role_split,
    manifest_rows,
    paired_group_bootstrap,
    row_key,
    reference_bucket,
    score_probabilities,
    text_background_features,
    text_sha256,
)
from b_text_detector.ood import connected_components, load_development_pool  # noqa: E402

TRAIN_REL = "artifacts/audited_train/splits/train.jsonl"
DEV_REL = "artifacts/audited_train/splits/dev.jsonl"
PROTOCOL_VERSION = "condition_baseline_v2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def subgroup_audit(rows: list[dict], y: np.ndarray, probabilities: dict[str, np.ndarray]) -> dict:
    """Descriptive groups only; source and script proxies never enter features."""
    groups = {"source": {}, "script_proxy": {}, "whitespace_length_bucket": {}}
    from b_text_detector.condition_baseline import text_background_features
    keys = []
    for row in rows:
        _, bucket, script = text_background_features(str(row["text"]))
        keys.append({"source": str(row.get("source", "unknown")),
                     "script_proxy": script, "whitespace_length_bucket": bucket})
    for group_name in groups:
        values = sorted({item[group_name] for item in keys})
        for value in values:
            mask = np.asarray([item[group_name] == value for item in keys])
            yp = np.asarray(y)[mask]
            if not len(yp):
                continue
            groups[group_name][value] = {"n": int(mask.sum()),
                "human_n": int(np.sum(yp == 0)), "machine_n": int(np.sum(yp == 1)), "models": {}}
            for model, p in probabilities.items():
                pred = (np.asarray(p)[mask] >= 0.5).astype(int)
                groups[group_name][value]["models"][model] = {
                    "macro_f1": float(f1_score(yp, pred, labels=[0, 1], average="macro", zero_division=0)),
                    "human_fpr": float(np.mean(pred[yp == 0])) if np.any(yp == 0) else None,
                    "machine_recall": float(np.mean(pred[yp == 1])) if np.any(yp == 1) else None,
                }
    return groups


def run_ood_protocols(manifest_dir: Path, out_dir: Path, source_splits: dict[str, list[dict]],
                      base_features: dict[str, dict[str, np.ndarray]],
                      base_buckets: dict[str, list[tuple[str, str]]]) -> dict:
    """Fit each reviewed OOD target only on its own R/F/S, then score dev/eval once."""
    outputs = {}
    source_lookup = {}
    for split, rows in source_splits.items():
        for i, row in enumerate(rows):
            source_lookup[(split, i)] = row
    for name in ("generator_davinci", "domain_wikihow"):
        files = {part: manifest_dir / f"{name}.{part}.roles.jsonl"
                 for part in ("train", "dev", "eval")}
        if any(not p.is_file() for p in files.values()):
            raise FileNotFoundError(f"{name}: role manifests missing; run prepare-manifests first")
        subsets = {}
        for part, path in files.items():
            manifest = read_jsonl(path)
            source_rows, source_indices, keys = [], [], []
            for row in manifest:
                split = row["original_split"]
                index = int(row["original_index"])
                source = source_lookup.get((split, index))
                if source is None:
                    raise ValueError(f"{name}/{part}: manifest points outside frozen source rows")
                if row["row_key"] != row_key(split, index, str(source["text"])):
                    raise ValueError(f"{name}/{part}: row key no longer matches frozen source")
                if int(row["label"]) != int(source["label"]):
                    raise ValueError(f"{name}/{part}: label changed since role manifest creation")
                source_rows.append(source)
                source_indices.append(index)
                keys.append(split)
            matrices = {set_name: np.asarray([
                base_features[split][set_name][idx]
                for split, idx in zip(keys, source_indices)], dtype=np.float32)
                for set_name in ("A", "B", "C_neutral", "C")}
            subsets[part] = {"manifest": manifest, "rows": source_rows, "matrices": matrices,
                             "labels": np.asarray([int(r["label"]) for r in manifest], dtype=np.int64),
                             "roles": [str(r["role"]) for r in manifest]}
        tr, dv, ev = (subsets[k] for k in ("train", "dev", "eval"))
        rebuilt_plan = make_role_split(tr["rows"])
        if rebuilt_plan["row_roles"] != tr["roles"]:
            raise ValueError(f"{name}: role assignments differ from deterministic protocol")
        if rebuilt_plan["row_groups"] != [int(r["group_id"]) for r in tr["manifest"]]:
            raise ValueError(f"{name}: training component IDs differ from lineage")
        _, eval_group_ids = connected_components(ev["rows"])
        if eval_group_ids != [int(r["group_id"]) for r in ev["manifest"]]:
            raise ValueError(f"{name}: evaluation component IDs differ from lineage")
        all_rows = tr["rows"] + dv["rows"] + ev["rows"]
        all_roles = tr["roles"] + ["OOD_DEV"] * len(dv["rows"]) + ["OOD_EVAL"] * len(ev["rows"])
        all_components, _ = connected_components(all_rows)
        for members in all_components:
            if len({all_roles[i] for i in members}) > 1:
                raise ValueError(f"{name}: connected article/body aliases cross train, dev, or eval roles")
        ref_ix = np.asarray([role == "R" and int(label) == 0 for role, label in zip(tr["roles"], tr["labels"])])
        if not ref_ix.any():
            raise ValueError(f"{name}: OOD R has no human reference rows")
        ref = fit_human_reference(tr["matrices"]["C"][ref_ix], [
            base_buckets[str(tr["manifest"][i]["original_split"])][int(tr["manifest"][i]["original_index"])]
            for i in np.flatnonzero(ref_ix)])
        for subset in (tr, dv, ev):
            buckets = [base_buckets[str(row["original_split"])][int(row["original_index"])]
                       for row in subset["manifest"]]
            subset["matrices"]["D_global"] = apply_human_reference(
                subset["matrices"]["C"], buckets, ref, use_global=True)
            subset["matrices"]["D"] = apply_human_reference(subset["matrices"]["C"], buckets, ref)
        y = tr["labels"]
        dy, ey = dv["labels"], ev["labels"]
        short = np.asarray([len(str(r["text"]).split()) < 100 for r in tr["rows"]])
        dshort = np.asarray([len(str(r["text"]).split()) < 100 for r in dv["rows"]])
        eshort = np.asarray([len(str(r["text"]).split()) < 100 for r in ev["rows"]])
        # One bounded selection/training pass; held eval is the evaluation target.
        outcome = fit_bounded_comparisons(tr["matrices"], ev["matrices"], y, tr["roles"], ey,
                                          train_short=short, dev_short=eshort)
        dev_probabilities = {set_name: model.predict_proba(dv["matrices"][set_name])[:, 1]
                             for set_name, model in outcome["models"].items()}
        eval_probabilities = {set_name: values["DEV"] for set_name, values in outcome["probabilities"].items()}
        group_ids = np.asarray([int(r["group_id"]) for r in ev["manifest"]])
        bootstrap = paired_group_bootstrap(ey, eval_probabilities, group_ids, replicates=200)
        family_out = out_dir / name
        family_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(family_out / "feature_arrays.npz", **{
            f"{part}_{set_name}": matrix
            for part, subset in (("train", tr), ("dev", dv), ("eval", ev))
            for set_name, matrix in subset["matrices"].items()})
        np.savez_compressed(family_out / "labels_and_roles.npz",
            train_labels=tr["labels"], dev_labels=dy, eval_labels=ey,
            train_roles=np.asarray(tr["roles"]),
            train_group_ids=np.asarray([int(r["group_id"]) for r in tr["manifest"]]))
        joblib.dump(ref, family_out / "human_reference.joblib")
        for set_name, model in outcome["models"].items():
            model_path = family_out / f"{set_name}_{outcome['by_comparison'][set_name]['selected_classifier']}.joblib"
            joblib.dump(model, model_path)
            loaded = joblib.load(model_path)
            if not np.allclose(model.predict_proba(ev["matrices"][set_name][:4]),
                               loaded.predict_proba(ev["matrices"][set_name][:4]), rtol=0, atol=0):
                raise ValueError(f"{name}/{set_name}: joblib reload changed predictions")
        for part, subset, probs in (("dev", dv, dev_probabilities), ("eval", ev, eval_probabilities)):
            atomic_jsonl(family_out / f"{part}_predictions.jsonl", [
                {"row_key": row["row_key"], "original_split": row["original_split"],
                 "original_index": int(row["original_index"]), "label": int(subset["labels"][i]),
                 **{f"{set_name}_probability": float(prob[i]) for set_name, prob in probs.items()}}
                for i, row in enumerate(subset["manifest"])])
        s_ix = np.flatnonzero(np.asarray(tr["roles"]) == "S")
        atomic_jsonl(family_out / "selection_predictions.jsonl", [
            {"row_key": tr["manifest"][int(ix)]["row_key"],
             "original_split": tr["manifest"][int(ix)]["original_split"],
             "original_index": int(tr["manifest"][int(ix)]["original_index"]),
             "label": int(y[ix]),
             **{f"{set_name}_{classifier}_probability": float(prob[j])
                for set_name, values in outcome["probabilities"].items()
                for classifier, prob in values["S_candidates"].items()}}
            for j, ix in enumerate(s_ix)])
        roc_csv = ["comparison,fpr,tpr,threshold"]
        calibration_csv = ["comparison,bin,n,mean_probability,fraction_machine"]
        scores_csv = ["comparison,classifier,n,macro_f1,short_macro_f1,combined_score,roc_auc,brier,log_loss,human_fpr,machine_recall"]
        for set_name, metrics in outcome["by_comparison"].items():
            for point in metrics["dev_metrics"]["roc"]:
                roc_csv.append(f"{set_name}_eval,{point['fpr']},{point['tpr']},{point['threshold']}")
            for classifier, scores in metrics["selection_scores"].items():
                scores_csv.append(",".join(str(x) for x in (set_name, classifier, scores["n"],
                    scores["macro_f1"], scores["short_macro_f1"], scores["combined_score"], scores["roc_auc"],
                    scores["brier"], scores["log_loss"], scores["human_fpr"], scores["machine_recall"])))
                for point in scores["roc"]:
                    roc_csv.append(f"{set_name}_{classifier}_selection,{point['fpr']},{point['tpr']},{point['threshold']}")
                for bin_row in metrics["selection_calibration"][classifier]:
                    calibration_csv.append(f"{set_name}_{classifier}_selection,{bin_row['bin']},{bin_row['n']},"
                                           f"{bin_row['mean_probability']},{bin_row['fraction_machine']}")
        for split_name, labels, probs in (("dev", dy, dev_probabilities), ("eval", ey, eval_probabilities)):
            for set_name, p in probs.items():
                for bin_row in calibration_rows(labels, p):
                    calibration_csv.append(f"{set_name}_{split_name},{bin_row['bin']},{bin_row['n']},"
                                           f"{bin_row['mean_probability']},{bin_row['fraction_machine']}")
        (family_out / "roc_points.csv").write_text("\n".join(roc_csv) + "\n", encoding="utf-8")
        (family_out / "calibration.csv").write_text("\n".join(calibration_csv) + "\n", encoding="utf-8")
        (family_out / "selection_scores.csv").write_text("\n".join(scores_csv) + "\n", encoding="utf-8")
        outputs[name] = {"selection_and_eval": outcome["by_comparison"],
            "dev_metrics": {set_name: score_probabilities(dy, prob, dshort)
                            for set_name, prob in dev_probabilities.items()},
            "dev_subgroup_audit": subgroup_audit(dv["rows"], dy, dev_probabilities),
            "eval_subgroup_audit": subgroup_audit(ev["rows"], ey, eval_probabilities),
            "eval_paired_group_bootstrap": {k: v for k, v in bootstrap.items() if k != "draws"},
            "eval_bootstrap_draws": bootstrap["draws"],
            "row_counts": {k: len(v["rows"]) for k, v in subsets.items()},
            "role_counts": {role: tr["roles"].count(role) for role in ("R", "F", "S")},
            "reference_fit_rows_human_only": True,
            "reference_fit_rows": int(ref_ix.sum()),
            "source_protocol_sha256": sha(ROOT / "artifacts/experiments/ood" / name / "manifest.json")}
    return outputs


def prepare_manifests(out_dir: Path) -> dict:
    train_path, dev_path = ROOT / TRAIN_REL, ROOT / DEV_REL
    for protected in ("locked_test.jsonl", "official_validation.jsonl", "validation.jsonl"):
        if protected in str(train_path) or protected in str(dev_path):
            raise ValueError("locked/official validation data is prohibited")
    rows = load_development_pool(train_path, dev_path)
    all_components, _ = connected_components(rows)
    cross_train_dev = sum(
        any(rows[i]["_original_split"] == "train" for i in group)
        and any(rows[i]["_original_split"] == "dev" for i in group)
        for group in all_components)
    if cross_train_dev:
        raise ValueError(f"frozen train/dev have {cross_train_dev} connected components crossing the split")
    train = [dict(r, _original_index=i) for i, r in enumerate(rows) if r["_original_split"] == "train"]
    dev = [dict(r, _original_index=i) for i, r in enumerate(
        r for r in rows if r["_original_split"] == "dev")]
    if len(train) != 85_369 or len(dev) != 13_387:
        raise ValueError(f"frozen train/dev row counts changed: {len(train)}/{len(dev)}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty manifest directory: {out_dir}")
    plan = make_role_split(train)
    train_manifest = manifest_rows(train, "train", row_roles=plan["row_roles"], group_ids=plan["row_groups"])
    dev_manifest = manifest_rows(dev, "dev")
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(out_dir / "iid_train.roles.jsonl", train_manifest)
    atomic_jsonl(out_dir / "iid_dev.roles.jsonl", dev_manifest)
    group_roles = {}
    for role, gids in plan["role_group_ids"].items():
        group_roles[role] = len(gids)
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": plan["seed"], "requested_fractions": plan["requested_fractions"],
        "role_row_counts": plan["role_row_counts"], "role_component_counts": group_roles,
        "component_cross_role_count": plan["cross_role_components"],
        "component_cross_train_dev_count": cross_train_dev,
        "R_has_human_reference_rows": True, "F_and_S_each_have_both_labels": True,
        "train_path": TRAIN_REL, "dev_path": DEV_REL,
        "train_sha256": sha(train_path), "dev_sha256": sha(dev_path),
        "train_manifest_sha256": sha(out_dir / "iid_train.roles.jsonl"),
        "dev_manifest_sha256": sha(out_dir / "iid_dev.roles.jsonl"),
        "train_rows": len(train), "dev_rows": len(dev), "locked_test_read": False,
        "role_source": "one deterministic whole-component assignment; no seed retry",
    }
    (out_dir / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Existing reviewed OOD partitions are mapped back to the original frozen
    # rows by split + label + body_hash before independently assigning R/F/S.
    source_rows = train + dev
    by_identity = {}
    for source_row in source_rows:
        identity = (source_row["_original_split"], int(source_row["label"]), source_row.get("body_hash"))
        if identity in by_identity:
            raise ValueError(f"frozen source identity is ambiguous: {identity[:2]}")
        by_identity[identity] = source_row
    ood_root = ROOT / "artifacts/experiments/ood"
    ood_audit = {}
    for family in ("generator_davinci", "domain_wikihow"):
        protocol = ood_root / family
        meta = json.loads((protocol / "manifest.json").read_text(encoding="utf-8"))
        if (meta.get("original_sources", {}).get("train_sha256") != sha(train_path)
                or meta.get("original_sources", {}).get("dev_sha256") != sha(dev_path)):
            raise ValueError(f"{family}: OOD protocol is not bound to the frozen source files")
        mapped = {}
        for split in ("train", "dev", "eval"):
            data_rows = read_jsonl(protocol / f"{split}.jsonl")
            lineage = read_jsonl(protocol / f"{split}.manifest.jsonl")
            if len(data_rows) != len(lineage):
                raise ValueError(f"{family}/{split}: OOD data and lineage counts differ")
            selected = []
            for row, lineage_row in zip(data_rows, lineage):
                identity = (str(lineage_row["original_split"]), int(lineage_row["label"]), lineage_row.get("body_hash"))
                source_row = by_identity.get(identity)
                if source_row is None or row.get("body_hash") != source_row.get("body_hash"):
                    raise ValueError(f"{family}/{split}: row cannot be mapped to frozen source by identity")
                selected.append(dict(source_row))
            mapped[split] = selected
        plan = make_role_split(mapped["train"])
        manifests = manifest_rows(mapped["train"], "train", row_roles=plan["row_roles"], group_ids=plan["row_groups"])
        atomic_jsonl(out_dir / f"{family}.train.roles.jsonl", manifests)
        for split in ("dev", "eval"):
            roles = [f"OOD_{split.upper()}"] * len(mapped[split])
            groups, ids = connected_components(mapped[split])
            atomic_jsonl(out_dir / f"{family}.{split}.roles.jsonl",
                         manifest_rows(mapped[split], split, row_roles=roles, group_ids=ids))
        ood_audit[family] = {"source_protocol_manifest_sha256": sha(protocol / "manifest.json"),
                             "train_sha256": sha(protocol / "train.jsonl"),
                             "role_row_counts": plan["role_row_counts"],
                             "role_component_counts": {role: len(gids) for role, gids in plan["role_group_ids"].items()},
                             "cross_role_components": plan["cross_role_components"],
                             "held_eval_is_separate": True}
    report["ood_role_audits"] = ood_audit
    (out_dir / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_release(probe_root: Path, release_path: Path, train_path: Path, dev_path: Path) -> dict:
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("released") is not True or release.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("formal CPU run is blocked: explicit parent FORMAL_RELEASE is missing or invalid")
    if Path(release.get("approved_run_root", "")).resolve() != probe_root.resolve():
        raise ValueError("FORMAL_RELEASE approved_run_root does not match --probe-root")
    if release.get("train_sha256") != sha(train_path) or release.get("dev_sha256") != sha(dev_path):
        raise ValueError("release marker does not bind the current frozen train/dev files")
    status = {}
    for split, expected_path in (("train", train_path), ("dev", dev_path)):
        d = probe_root / split
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        run_status = json.loads((d / "run_status.json").read_text(encoding="utf-8"))
        source_rows = read_jsonl(expected_path)
        if (run_status.get("state") != "complete" or run_status.get("failed") is not False
                or run_status.get("split") != split
                or int(run_status.get("completed", -1)) != len(source_rows)
                or int(run_status.get("total", -1)) != len(source_rows)):
            raise ValueError(f"{split} condition-probe extraction is incomplete")
        if manifest.get("split") != split or manifest.get("input_sha256") != sha(expected_path):
            raise ValueError(f"{split} condition-probe manifest does not bind the frozen input")
        if int(manifest.get("selection", {}).get("selected_rows", -1)) != len(source_rows):
            raise ValueError(f"{split} probe manifest does not include every frozen source row")
        if int(manifest.get("body_budget_tokens", -1)) != 448:
            raise ValueError(f"{split} condition-probe manifest is not the required 448-token run")
        if (int(manifest.get("scoring_batch_size", -1)) != 1
                or manifest.get("scoring_policy") != "single_article_forward_per_condition_no_mixed_padding"):
            raise ValueError(f"{split} condition-probe manifest is not the required singleton scoring run")
        if (set(manifest.get("templates", {})) != {"raw", "neutral", "human_hypothesis", "machine_hypothesis"}
                or manifest.get("model") != "Qwen/Qwen3-4B"
                or manifest.get("model_revision") != "1cfa9a7208912126459214e8b04321603b3df60c"
                or manifest.get("adapter") is not None):
            raise ValueError(f"{split} probe does not use the frozen four-condition base-model contract")
        prefix_lengths = manifest.get("prefix_lengths", {})
        if len({prefix_lengths.get(k) for k in ("neutral", "human_hypothesis", "machine_hypothesis")}) != 1:
            raise ValueError(f"{split} neutral/human/machine prefix token counts differ")
        status[split] = {"manifest": manifest, "run_status": run_status,
                         "rows_path": d / "rows.jsonl"}
    for key in ("train_manifest_sha256", "dev_manifest_sha256"):
        split = key.removesuffix("_manifest_sha256")
        actual = sha(probe_root / split / "manifest.json")
        if release.get(key) != actual:
            raise ValueError(f"FORMAL_RELEASE {key} does not match probe manifest")
    return status


def run_formal(probe_root: Path, out_dir: Path, release_path: Path, manifest_dir: Path) -> dict:
    # Fail before reading the large probes or fitting any model when a previous
    # run already left outputs behind. The supervisor adds an OS-level lock.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty formal output: {out_dir}")
    train_path, dev_path = ROOT / TRAIN_REL, ROOT / DEV_REL
    status = verify_release(probe_root, release_path, train_path, dev_path)
    train = [dict(r, _original_split="train", _original_index=i) for i, r in enumerate(read_jsonl(train_path))]
    dev = [dict(r, _original_split="dev", _original_index=i) for i, r in enumerate(read_jsonl(dev_path))]
    split_plan = make_role_split(train)
    train_role_manifest = read_jsonl(manifest_dir / "iid_train.roles.jsonl")
    dev_role_manifest = read_jsonl(manifest_dir / "iid_dev.roles.jsonl")
    if len(train_role_manifest) != len(train) or len(dev_role_manifest) != len(dev):
        raise ValueError("prepared IID role manifests do not cover frozen train/dev exactly")
    expected_roles = manifest_rows(train, "train", row_roles=split_plan["row_roles"], group_ids=split_plan["row_groups"])
    expected_dev = manifest_rows(dev, "dev")
    for actual, expected in zip(train_role_manifest, expected_roles):
        if any(actual.get(k) != expected.get(k) for k in ("row_key", "text_sha256", "original_index", "group_id", "role", "label")):
            raise ValueError("prepared IID train role manifest differs from deterministic split or frozen data")
    for actual, expected in zip(dev_role_manifest, expected_dev):
        if any(actual.get(k) != expected.get(k) for k in ("row_key", "text_sha256", "original_index", "role", "label")):
            raise ValueError("prepared IID dev role manifest differs from frozen data")
    train_base, feature_names, buckets, _ = build_feature_matrices_stream(
        train, status["train"]["rows_path"], "train")
    dev_base, dev_names, dev_buckets, _ = build_feature_matrices_stream(
        dev, status["dev"]["rows_path"], "dev")
    if feature_names != dev_names:
        raise ValueError("train/dev condition-probe feature schema differs")
    y = np.asarray([int(r["label"]) for r in train], dtype=np.int64)
    dev_y = np.asarray([int(r["label"]) for r in dev], dtype=np.int64)
    train_short = np.asarray([len(str(r["text"]).split()) < 100 for r in train], dtype=bool)
    dev_short = np.asarray([len(str(r["text"]).split()) < 100 for r in dev], dtype=bool)
    row_roles = [str(row["role"]) for row in train_role_manifest]
    r_ix = np.asarray([(role == "R" and int(label) == 0)
                       for role, label in zip(row_roles, y)])
    if not r_ix.any() or set(y[r_ix]) != {0}:
        raise ValueError("R has no human rows for reference fitting")
    ref = fit_human_reference(train_base["C"][r_ix], [buckets[i] for i in np.flatnonzero(r_ix)])
    train_features = {**train_base,
                      "D_global": apply_human_reference(train_base["C"], buckets, ref, use_global=True),
                      "D": apply_human_reference(train_base["C"], buckets, ref)}
    dev_features = {**dev_base,
                    "D_global": apply_human_reference(dev_base["C"], dev_buckets, ref, use_global=True),
                    "D": apply_human_reference(dev_base["C"], dev_buckets, ref)}
    result = fit_bounded_comparisons(train_features, dev_features, y, row_roles, dev_y,
                                    train_short=train_short, dev_short=dev_short)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "configs/condition_baseline.json", out_dir / "config.snapshot.json")
    shutil.copyfile(ROOT / "reports/CONDITION_BASELINE_PROTOCOL.md", out_dir / "protocol.snapshot.md")
    np.savez_compressed(out_dir / "feature_arrays.npz", **{
        f"train_{k}": v for k, v in train_features.items()}, **{f"dev_{k}": v for k, v in dev_features.items()})
    np.savez_compressed(out_dir / "labels_and_roles.npz", train_labels=y, dev_labels=dev_y,
                        train_roles=np.asarray(split_plan["row_roles"]),
                        train_group_ids=np.asarray(split_plan["row_groups"]))
    joblib.dump({"reference": ref, "feature_names": feature_names}, out_dir / "feature_reference.joblib")
    for set_name, model in result["models"].items():
        model_path = out_dir / f"{set_name}_{result['by_comparison'][set_name]['selected_classifier']}.joblib"
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
        check_ix = np.arange(min(8, len(dev_features[set_name])))
        before = model.predict_proba(dev_features[set_name][check_ix])
        after = reloaded.predict_proba(dev_features[set_name][check_ix])
        if not np.allclose(before, after, rtol=0, atol=0):
            raise ValueError(f"joblib reload changed predictions for {set_name}")
    train_man = train_role_manifest
    dev_man = dev_role_manifest
    atomic_jsonl(out_dir / "iid_train.roles.jsonl", train_man)
    atomic_jsonl(out_dir / "iid_dev.roles.jsonl", dev_man)
    _, dev_groups = connected_components(dev)
    boot = paired_group_bootstrap(dev_y, {k: v["DEV"] for k, v in result["probabilities"].items()},
                                  np.asarray(dev_groups), replicates=200)
    ood_results = run_ood_protocols(manifest_dir, out_dir, {"train": train, "dev": dev},
                                    {"train": train_features, "dev": dev_features},
                                    {"train": buckets, "dev": dev_buckets})
    prediction_rows = []
    for i, meta in enumerate(dev_man):
        record = {"row_key": meta["row_key"], "original_index": i, "label": int(dev_y[i])}
        for name, vectors in result["probabilities"].items():
            record[f"{name}_probability"] = float(vectors["DEV"][i])
        prediction_rows.append(record)
    atomic_jsonl(out_dir / "dev_predictions.jsonl", prediction_rows)
    atomic_jsonl(out_dir / "selection_predictions.jsonl", [
        {"row_key": train_man[int(ix)]["row_key"], "original_index": int(ix), "label": int(y[ix]),
         **{f"{name}_{classifier}_probability": float(prob[j])
            for name, vectors in result["probabilities"].items()
            for classifier, prob in vectors["S_candidates"].items()}}
        for j, ix in enumerate(np.flatnonzero(np.asarray(row_roles) == "S"))
    ])
    calibration = {}
    roc_csv = ["comparison,fpr,tpr,threshold"]
    calibration_csv = ["comparison,bin,n,mean_probability,fraction_machine"]
    scores_csv = ["comparison,classifier,n,macro_f1,short_macro_f1,combined_score,roc_auc,brier,log_loss,human_fpr,machine_recall"]
    for name, vectors in result["probabilities"].items():
        p = vectors["DEV"]
        calibration[name] = calibration_rows(dev_y, p)
        metrics = result["by_comparison"][name]["dev_metrics"]
        for point in metrics["roc"]:
            roc_csv.append(f"{name},{point['fpr']},{point['tpr']},{point['threshold']}")
        for classifier, selection_metrics in result["by_comparison"][name]["selection_scores"].items():
            scores_csv.append(",".join(str(x) for x in (name, classifier, selection_metrics["n"],
                selection_metrics["macro_f1"], selection_metrics["short_macro_f1"],
                selection_metrics["combined_score"], selection_metrics["roc_auc"], selection_metrics["brier"],
                selection_metrics["log_loss"], selection_metrics["human_fpr"], selection_metrics["machine_recall"])))
            for point in selection_metrics["roc"]:
                roc_csv.append(f"{name}_{classifier}_selection,{point['fpr']},{point['tpr']},{point['threshold']}")
            for b in result["by_comparison"][name]["selection_calibration"][classifier]:
                calibration_csv.append(f"{name}_{classifier}_selection,{b['bin']},{b['n']},"
                                       f"{b['mean_probability']},{b['fraction_machine']}")
        for b in calibration[name]:
            calibration_csv.append(f"{name},{b['bin']},{b['n']},{b['mean_probability']},{b['fraction_machine']}")
    (out_dir / "roc_points.csv").write_text("\n".join(roc_csv) + "\n", encoding="utf-8")
    (out_dir / "calibration.csv").write_text("\n".join(calibration_csv) + "\n", encoding="utf-8")
    (out_dir / "selection_scores.csv").write_text("\n".join(scores_csv) + "\n", encoding="utf-8")
    report = {
        "protocol_version": PROTOCOL_VERSION, "run_kind": "formal", "feature_names": feature_names,
        "source_hashes": {TRAIN_REL: sha(train_path), DEV_REL: sha(dev_path)},
        "role_manifest_hashes": {path.name: sha(path) for path in sorted(manifest_dir.glob("*.roles.jsonl"))},
        "split_audit_sha256": sha(manifest_dir / "split_audit.json"),
        "probe_manifest_hashes": {s: sha(probe_root / s / "manifest.json") for s in ("train", "dev")},
        "probe_rows_sha256": {s: sha(status[s]["rows_path"]) for s in ("train", "dev")},
        "release_marker_sha256": sha(release_path),
        "role_assignment": {"seed": split_plan["seed"],
            "row_counts": {role: row_roles.count(role) for role in ("R", "F", "S")},
            "component_count": len(split_plan["components"]), "cross_role_components": 0,
            "R_reference_fit_rows_human_only": True,
            "R_role_rows": row_roles.count("R"), "R_reference_fit_rows": int(r_ix.sum()),
            "S_has_both_labels": True, "F_has_both_labels": True},
        "model_config": {"LR": {"C": 1.0, "max_iter": 1000},
            "HGB": {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 15,
                    "l2_regularization": 1.0, "min_samples_leaf": 30, "early_stopping": False}},
        "supervision_budget_note": "D and D_global use Rhuman reference statistics; C does not. D_global shares the same Rhuman rows with D.",
        "selection_and_dev": result["by_comparison"], "calibration": calibration,
        "subgroup_audit": subgroup_audit(dev, dev_y, {k: v["DEV"] for k, v in result["probabilities"].items()}),
        "ood_protocol_results": ood_results,
        "paired_group_bootstrap": {k: v for k, v in boot.items() if k != "draws"},
        "bootstrap_draws": boot["draws"],
        "run_metadata": {"python": platform.python_version(), "dependencies": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib", "threadpoolctl")},
            "cpu_threads": 2, "seed": 20261019,
            "config_sha256": sha(ROOT / "configs/condition_baseline.json"),
            "module_sha256": sha(ROOT / "src/b_text_detector/condition_baseline.py"),
            "runner_sha256": sha(Path(__file__))},
        "formal_status": {split: {**value, "rows_path": str(value["rows_path"])}
                          for split, value in status.items()},
        "prohibited_inputs": ["full text", "source", "generator family", "language", "ID metadata", "Qwen fusion"],
        "text_exported": False,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    (out_dir / "run_status.json").write_text(json.dumps({"state": "complete", "failed": False,
        "protocol_version": PROTOCOL_VERSION, "probe_root": str(probe_root),
        "iid_train_rows": len(train), "iid_dev_rows": len(dev),
        "ood_protocols": sorted(ood_results)}, indent=2) + "\n", encoding="utf-8")
    (out_dir / "run.log").write_text(
        f"protocol={PROTOCOL_VERSION}\nprobe_root={probe_root}\nrelease_marker_sha256={sha(release_path)}\n"
        f"train_rows={len(train)} dev_rows={len(dev)}\n"
        "formal condition baseline completed; see report.json and selection_scores.csv.\n",
        encoding="utf-8")
    return report


def run_synthetic_fixture(out_dir: Path) -> dict:
    """Small non-empirical end-to-end exercise of transforms and CPU fitting."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite fixture output: {out_dir}")
    train, dev, train_probe, dev_probe = [], [], [], []
    def make_probe(index: int, label: int, offset: float) -> dict:
        base = float(label) * 0.4 + offset
        summary = {"scored_token_count": 20, "mean": base, "std": 0.2,
                   "q10": base - 0.2, "q25": base - 0.1, "median": base,
                   "q75": base + 0.1, "q90": base + 0.2,
                   "window16_means": [{"mean": base + j * 0.001, "count": 16,
                                       "start_body_offset": j * 16} for j in range(2)]}
        return {"conditions": {name: dict(summary) for name in
                ("raw", "neutral", "human_hypothesis", "machine_hypothesis")},
                "pairwise_surprise_deltas": {name: dict(summary) for name in
                ("neutral_minus_raw", "human_minus_neutral", "machine_minus_neutral", "machine_minus_human")}}
    for i in range(240):
        label = i % 2
        row = {"text": f"Synthetic fixture {i} " + ("ordinary words " * 12), "label": label,
               "article_id": f"fixture-{i}", "body_hash": f"fixture-body-{i}",
               "duplicate_article_ids": [], "alias_body_hashes": [], "_original_split": "train"}
        train.append(row)
        probe = make_probe(i, label, (i % 5) * 0.01)
        probe.update({"row_key": row_key("train", i, row["text"]), "text_sha256": text_sha256(row["text"]),
                      "original_index": i, "label": label})
        train_probe.append(probe)
    for i in range(40):
        label = i % 2
        row = {"text": f"Synthetic dev fixture {i} " + ("ordinary words " * 12), "label": label,
               "article_id": f"fixture-dev-{i}", "body_hash": f"fixture-dev-body-{i}",
               "duplicate_article_ids": [], "alias_body_hashes": [], "_original_split": "dev"}
        dev.append(row)
        probe = make_probe(i, label, (i % 5) * 0.01)
        probe.update({"row_key": row_key("dev", i, row["text"]), "text_sha256": text_sha256(row["text"]),
                      "original_index": i, "label": label})
        dev_probe.append(probe)
    split_plan = make_role_split(train)
    train_base, names, buckets, _ = build_feature_matrices(train, train_probe)
    dev_base, dev_names, dev_buckets, _ = build_feature_matrices(dev, dev_probe)
    if names != dev_names:
        raise AssertionError("synthetic feature schema drift")
    y = np.asarray([r["label"] for r in train], dtype=np.int64)
    dy = np.asarray([r["label"] for r in dev], dtype=np.int64)
    train_short = np.asarray([len(r["text"].split()) < 100 for r in train], dtype=bool)
    dev_short = np.asarray([len(r["text"].split()) < 100 for r in dev], dtype=bool)
    ref_ix = np.asarray([role == "R" and int(label) == 0
                         for role, label in zip(split_plan["row_roles"], y)])
    reference = fit_human_reference(train_base["C"][ref_ix], [buckets[i] for i in np.flatnonzero(ref_ix)])
    train_features = {**train_base,
                      "D_global": apply_human_reference(train_base["C"], buckets, reference, use_global=True),
                      "D": apply_human_reference(train_base["C"], buckets, reference)}
    dev_features = {**dev_base,
                    "D_global": apply_human_reference(dev_base["C"], dev_buckets, reference, use_global=True),
                    "D": apply_human_reference(dev_base["C"], dev_buckets, reference)}
    outcome = fit_bounded_comparisons(train_features, dev_features, y, split_plan["row_roles"], dy,
                                     train_short=train_short, dev_short=dev_short)
    out_dir.mkdir(parents=True, exist_ok=True)
    reload_checks = {}
    for set_name, model in outcome["models"].items():
        model_path = out_dir / f"{set_name}_{outcome['by_comparison'][set_name]['selected_classifier']}.joblib"
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
        ix = np.arange(min(4, len(dev_features[set_name])))
        reload_checks[set_name] = bool(np.allclose(model.predict_proba(dev_features[set_name][ix]),
                                                   reloaded.predict_proba(dev_features[set_name][ix]),
                                                   rtol=0, atol=0))
    report = {"protocol_version": PROTOCOL_VERSION, "run_kind": "synthetic_engineering_fixture",
              "metrics_are_synthetic_and_not_empirical": True,
              "row_counts": {"train": len(train), "dev": len(dev), **split_plan["role_row_counts"]},
        "feature_dimensions": {name: int(x.shape[1]) for name, x in train_features.items()},
              "selection_scores": {name: value["selection_scores"]
                                   for name, value in outcome["by_comparison"].items()},
              "selected_models": {name: value["selected_classifier"]
                                  for name, value in outcome["by_comparison"].items()},
              "joblib_save_reload_verified": reload_checks,
              "full_text_exported": False}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-manifests")
    prep.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/condition_baseline/manifests")
    run = sub.add_parser("run")
    run.add_argument("--probe-root", type=Path, required=True)
    run.add_argument("--release-marker", type=Path, default=ROOT / "artifacts/condition_baseline/FORMAL_RELEASE.json")
    run.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/condition_baseline/formal")
    run.add_argument("--manifest-dir", type=Path, default=ROOT / "artifacts/condition_baseline/manifests")
    fixture = sub.add_parser("synthetic-fixture")
    fixture.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/condition_baseline/synthetic_engineering_fixture")
    args = parser.parse_args()
    if args.command == "prepare-manifests":
        report = prepare_manifests(args.output_dir)
    elif args.command == "synthetic-fixture":
        report = run_synthetic_fixture(args.output_dir)
    else:
        with threadpool_limits(limits=2):
            report = run_formal(args.probe_root, args.output_dir, args.release_marker, args.manifest_dir)
    summary_keys = ("run_kind", "role_row_counts", "selection_and_dev", "component_cross_role_count",
                    "row_counts", "feature_dimensions", "selected_models", "joblib_save_reload_verified")
    print(json.dumps({k: report.get(k) for k in summary_keys if k in report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
