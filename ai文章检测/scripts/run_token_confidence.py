#!/usr/bin/env python
"""Pilot/full extraction and train-only standardized LR for P1-01."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from b_text_detector.metrics import classification_metrics
from b_text_detector.token_confidence import (
    DEFAULT_REVISION, FEATURE_NAMES, cache_contract, extract_split,
    read_rows, sha256_file, tokenizer_fingerprint, score_batch_chunked,
    _read_jsonl,
)


def _length_features(rows):
    # A deliberately simple shortcut baseline, fit/scaled on train only.
    return np.asarray([[math.log1p(len(r["text"])), math.log1p(len(r["text"].split()))]
                       for r in rows], dtype=np.float64)


def _report(name, train_rows, dev_rows, train_feat, dev_feat, output, *, run_metadata):
    xtr = np.asarray([[float(r[k]) for k in FEATURE_NAMES] for r in train_feat], dtype=np.float64)
    xdv = np.asarray([[float(r[k]) for k in FEATURE_NAMES] for r in dev_feat], dtype=np.float64)
    ytr = np.asarray([r["label"] for r in train_rows], dtype=int)
    ydv = np.asarray([r["label"] for r in dev_rows], dtype=int)
    if len(np.unique(ytr)) != 2:
        raise ValueError("training subset must contain both labels")
    models = {
        "token_surprise": (xtr, xdv, FEATURE_NAMES),
        "surprise_without_length": (
            xtr[:, [i for i, n in enumerate(FEATURE_NAMES) if n != "valid_token_count"]],
            xdv[:, [i for i, n in enumerate(FEATURE_NAMES) if n != "valid_token_count"]],
            tuple(n for n in FEATURE_NAMES if n != "valid_token_count")),
        "length_only": (_length_features(train_rows), _length_features(dev_rows), ("log_chars", "log_whitespace_words")),
    }
    groups = {"short": np.asarray([len(r["text"].split()) < 100 for r in dev_rows]),
              "long": np.asarray([len(r["text"].split()) >= 100 for r in dev_rows])}
    result = {"threshold": 0.5, "fit_scope": "train only", "length_short_definition": "whitespace words < 100",
              "models": {}, "feature_names": list(FEATURE_NAMES),
              "run_metadata": run_metadata,
              "token_probability_note": "true-token surprise summaries, not full-distribution entropy; the full token_surprise model includes valid_token_count, while surprise_without_length omits that explicit count but can still reflect text length"}
    predictions = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for key, (tr, dv, names) in models.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=None, random_state=17))
        clf.fit(tr, ytr)
        probs = clf.predict_proba(dv)[:, 1]
        joblib.dump(clf, output.parent / f"{key}.joblib")
        metrics = classification_metrics(ydv, probs, groups["short"], threshold=0.5, rows=dev_rows)
        pred = probs >= 0.5
        metrics["classification_report"] = classification_report(ydv, pred.astype(int), labels=[0, 1], output_dict=True, zero_division=0)
        metrics["dev_confusion_matrix_labels_0_1"] = confusion_matrix(ydv, pred, labels=[0, 1]).tolist()
        metrics["dev_by_source"] = {}
        for source in sorted({str(r.get("source", "unknown")) for r in dev_rows}):
            ix = np.asarray([r.get("source", "unknown") == source for r in dev_rows])
            metrics["dev_by_source"][source] = {"n": int(ix.sum()), "macro_f1": float(f1_score(ydv[ix], pred[ix], labels=[0, 1], average="macro", zero_division=0)),
                                                 "human_fpr": float(((pred[ix]) & (ydv[ix] == 0)).sum() / max(1, (ydv[ix] == 0).sum())),
                                                 "machine_recall": float(((pred[ix]) & (ydv[ix] == 1)).sum() / max(1, (ydv[ix] == 1).sum()))}
        metrics["dev_by_length"] = {bucket: {"n": int(mask.sum()), "macro_f1": float(f1_score(ydv[mask], pred[mask], labels=[0, 1], average="macro", zero_division=0)) if mask.any() else None}
                                    for bucket, mask in groups.items()}
        result["models"][key] = {"feature_names": list(names), **metrics}
        for i, row in enumerate(dev_rows):
            if len(predictions) <= i:
                predictions.append({"row_key": row["row_key"], "label": int(row["label"]),
                                    "source": row.get("source", "unknown"),
                                    "short": bool(groups["short"][i])})
            predictions[i][f"{key}_probability"] = float(probs[i])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output.parent / "dev_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def _make_contracts(cfg, splits, tokenizer):
    contracts = {}
    for split in splits:
        path = ROOT / cfg[f"{split}_path"]
        contracts[split] = cache_contract(
            split=split, input_hash=sha256_file(path), model_name=cfg["model_name"],
            revision=cfg["revision"], tokenizer_name=f"{cfg['model_name']}@{cfg['revision']}",
            tokenizer_hash=tokenizer_fingerprint(tokenizer), precision="bf16", max_length=512)
    return contracts


def _validated_features(outdir, split, rows, contract):
    meta_path = outdir / f"{split}.meta.json"
    feature_path = outdir / f"{split}.features.jsonl"
    if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != contract:
        raise ValueError(f"missing or stale {split} feature contract")
    cached = _read_jsonl(feature_path, repair_partial_tail=True)
    expected = {r["row_key"] for r in rows}
    keys = [r.get("row_key") for r in cached]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"{split} cache rows do not exactly match current data selection")
    for row in cached:
        missing = set(FEATURE_NAMES) - set(row)
        if missing or not np.isfinite([float(row[k]) for k in FEATURE_NAMES]).all():
            raise ValueError(f"invalid/non-finite features in {split} cache")
    by_key = {r["row_key"]: r for r in cached}
    return [by_key[r["row_key"]] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/token_confidence.json"))
    ap.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    ap.add_argument("--limit", type=int, help="pilot cap per split; omit only after parent acceptance")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--report", action="store_true", help="fit train-only LR and report dev from cached rows")
    ap.add_argument("--skip-model-load", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--verify-padding-invariance", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["revision"] != DEFAULT_REVISION or cfg["precision"] != "bfloat16" or cfg["max_length"] != 512:
        raise ValueError("P1 contract requires pinned revision, bf16 and max_length=512")
    frozen_paths = {"train": ROOT / "artifacts/audited_train/splits/train.jsonl",
                    "dev": ROOT / "artifacts/audited_train/splits/dev.jsonl"}
    for split in args.splits:
        configured = (ROOT / cfg[f"{split}_path"]).resolve()
        if configured != frozen_paths[split].resolve():
            raise ValueError(f"{split} path must resolve to the frozen audited {split}.jsonl")
    data = {}
    for split in args.splits:
        path = ROOT / cfg[f"{split}_path"]
        data[split] = read_rows(path)
        if args.limit:
            # Uniform deterministic sampling across the frozen split gives a
            # more useful cost pilot than taking only the first source block.
            if args.limit < len(data[split]):
                indexes = np.linspace(0, len(data[split]) - 1, num=args.limit, dtype=int)
                data[split] = [data[split][i] for i in indexes]
            print(json.dumps({"event": "pilot_selection", "split": split,
                              "strategy": "uniform_indices", "examples": len(data[split])}, ensure_ascii=False), flush=True)
    outdir = ROOT / cfg["output_dir"]
    tokenizer = None
    contracts = None
    if not args.skip_model_load:
        if not torch.cuda.is_available():
            raise RuntimeError("reference model extraction requires CUDA")
        start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["revision"], local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], revision=cfg["revision"], dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
            attn_implementation="sdpa",
        )
        model.eval()
        if args.verify_padding_invariance:
            # Select two different-length train texts only; compare one item's
            # feature vector alone and with the other as right-padding context.
            sample = data.get("train", data[args.splits[0]])
            candidates = sample[: min(32, len(sample))]
            encoded_candidates = tokenizer([r["text"] for r in candidates], truncation=True,
                                           max_length=512, padding=False, return_tensors=None)
            lengths = [len(ids) for ids in encoded_candidates["input_ids"]]
            shortest = min(range(len(lengths)), key=lambda k: lengths[k]) if lengths else None
            longest = max(range(len(lengths)), key=lambda k: lengths[k]) if lengths else None
            pair = (shortest, longest) if shortest is not None and lengths[shortest] != lengths[longest] else None
            if pair is None:
                raise ValueError("padding invariance check needs two different token lengths")
            i, j = pair
            special = tokenizer.all_special_ids
            single = tokenizer([candidates[i]["text"]], truncation=True, max_length=512,
                               padding=True, return_tensors="pt")
            mixed = tokenizer([candidates[i]["text"], candidates[j]["text"]], truncation=True,
                              max_length=512, padding=True, return_tensors="pt")
            dev = next(model.parameters()).device
            solo_feat = score_batch_chunked(model, single["input_ids"].to(dev),
                                            single["attention_mask"].to(dev), special)[0]
            mixed_feat = score_batch_chunked(model, mixed["input_ids"].to(dev),
                                             mixed["attention_mask"].to(dev), special)[0]
            diff = {k: abs(float(solo_feat[k]) - float(mixed_feat[k])) for k in FEATURE_NAMES}
            print(json.dumps({"event": "padding_invariance", "lengths": [lengths[i], lengths[j]],
                              "max_abs_feature_delta": max(diff.values()), "deltas": diff}, ensure_ascii=False), flush=True)
            stable_fields = ("surprise_mean", "surprise_std", "low_probability_fraction",
                             "adjacent_abs_delta_mean", "valid_token_count")
            stable_delta = max(diff[k] for k in stable_fields)
            print(json.dumps({"event": "padding_invariance_result",
                              "stable_fields_max_abs_delta": stable_delta,
                              "quantile_max_abs_delta": max(diff[k] for k in
                                  ("surprise_q10", "surprise_q25", "surprise_median", "surprise_q75", "surprise_q90")),
                              "tolerance": 2e-2, "pass": stable_delta <= 2e-2}, ensure_ascii=False), flush=True)
            if stable_delta > 2e-2:
                raise RuntimeError("right-padding invariance failed on stable summary statistics")
            outdir.mkdir(parents=True, exist_ok=True)
            padding_result = {"sequence_lengths": [lengths[i], lengths[j]],
                              "single_features": solo_feat, "mixed_batch_features": mixed_feat,
                              "absolute_deltas": diff, "stable_fields_max_abs_delta": stable_delta,
                              "quantile_max_abs_delta": max(diff[k] for k in
                                  ("surprise_q10", "surprise_q25", "surprise_median", "surprise_q75", "surprise_q90")),
                              "stable_tolerance": 2e-2, "passed": True,
                              "model_revision": cfg["revision"], "precision": "bf16"}
            (outdir / "padding_check.json").write_text(
                json.dumps(padding_result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"event": "model_loaded", "seconds": time.monotonic() - start,
                          "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device)}, ensure_ascii=False), flush=True)
        contracts = _make_contracts(cfg, args.splits, tokenizer)
        for split in args.splits:
            contract = contracts[split]
            extract_split(data[split], split=split, output_dir=outdir, tokenizer=tokenizer,
                          model=model, contract=contract,
                          batch_size=args.batch_size or cfg["batch_size"], limit=args.limit)
        del model
    if args.report:
        if set(args.splits) != {"train", "dev"}:
            raise ValueError("report requires both train and dev")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["revision"], local_files_only=True)
            contracts = _make_contracts(cfg, args.splits, tokenizer)
        feats = {}
        for split in ("train", "dev"):
            feats[split] = _validated_features(outdir, split, data[split], contracts[split])
        config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
        metadata = {"config_sha256": config_hash,
                    "data_sha256": {s: contracts[s]["input_sha256"] for s in ("train", "dev")},
                    "feature_contracts": contracts,
                    "dependencies": {name: importlib.metadata.version(name) for name in
                                     ("numpy", "scikit-learn", "joblib", "torch", "transformers")},
                    "python": sys.version.split()[0], "fit_split": "train", "evaluation_split": "dev"}
        report_path = outdir / ("pilot_report.json" if args.limit else "report.json")
        _report("pilot" if args.limit else "full", data["train"], data["dev"], feats["train"], feats["dev"], report_path,
                run_metadata=metadata)


if __name__ == "__main__":
    main()
