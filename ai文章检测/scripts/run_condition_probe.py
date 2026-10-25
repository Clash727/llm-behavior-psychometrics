#!/usr/bin/env python
"""Extract four fixed prefix-condition likelihood traces from frozen Qwen."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from b_text_detector.condition_probe import (
    CONDITIONS, MODEL_NAME, MODEL_REVISION, PAIRWISE, TEMPLATES, atomic_json,
    RunLock, encode_body, encode_contract, extract_split, make_manifest,
    read_rows, score_condition_batch, sha256_file, validate_split_path,
)


def select_train_pilot(rows: list[dict], limit: int | None, split: str) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    if split != "train":
        raise ValueError("pilot row selection is train-only; dev must use its full frozen split")
    if limit < 1:
        raise ValueError("limit must be positive")
    # Deterministic spread across the frozen train file; preserves original row IDs.
    indexes = np.linspace(0, len(rows) - 1, num=limit, dtype=int)
    return [rows[int(i)] for i in indexes]


def script_bucket(text: str) -> str:
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            key = "latin"
        elif any(x in name for x in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            key = "cjk"
        elif "CYRILLIC" in name:
            key = "cyrillic"
        elif "ARABIC" in name:
            key = "arabic"
        elif "GREEK" in name:
            key = "greek"
        elif "HEBREW" in name:
            key = "hebrew"
        elif "DEVANAGARI" in name:
            key = "devanagari"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def select_repeat_examples(rows: list[dict], tokenizer, count_per_script: int = 2):
    groups: dict[str, list[tuple[int, list[int]]]] = {}
    for i, row in enumerate(rows):
        bucket = script_bucket(row["text"])
        groups.setdefault(bucket, []).append((i, encode_body(tokenizer, row["text"])))
    required = ("latin", "cyrillic", "arabic", "cjk")
    absent = [x for x in required if x not in groups]
    if absent:
        raise ValueError(f"singleton repeat sample lacks required scripts: {absent}")
    selected = []
    for bucket in required:
        candidates = sorted(groups[bucket], key=lambda x: (len(x[1]), x[0]))
        for item in (candidates[0], candidates[-1]):
            if item[0] not in {x[0] for x in selected}:
                selected.append((item[0], item[1], bucket))
    if len(selected) != len(required) * count_per_script:
        raise ValueError("singleton repeat sample did not select two rows per script")
    return selected


def verify_singleton_repeat(model, tokenizer, rows: list[dict], prefix_ids: dict[str, list[int]]):
    selected = select_repeat_examples(rows, tokenizer)
    indexes = [x[0] for x in selected]
    device = next(model.parameters()).device
    traces: dict[int, dict[str, np.ndarray]] = {}
    start = time.monotonic()
    for order_name, order in (("forward", indexes), ("reverse", list(reversed(indexes)))):
        for index in order:
            body = selected[indexes.index(index)][1]
            current = {}
            for condition in CONDITIONS:
                current[condition] = score_condition_batch(
                    model, [body], prefix_ids[condition],
                    special_token_ids=tokenizer.all_special_ids, device=device)[0]
            if order_name == "forward":
                traces[index] = current
            else:
                first = traces[index]
                paired = {name: first[left] - first[right] for name, (left, right) in PAIRWISE.items()}
                repeated = {name: current[left] - current[right] for name, (left, right) in PAIRWISE.items()}
                records = {}
                all_exact = True
                max_abs = 0.0
                def digest(array: np.ndarray) -> str:
                    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
                for condition in CONDITIONS:
                    a, b = first[condition], current[condition]
                    exact = bool(np.array_equal(a, b, equal_nan=True))
                    finite = np.isfinite(a) & np.isfinite(b)
                    error = np.abs(a[finite] - b[finite])
                    max_condition = float(error.max(initial=0.0))
                    max_abs = max(max_abs, max_condition)
                    all_exact &= exact
                    records[condition] = {"exact": exact, "max_abs_delta": max_condition,
                                          "first_sha256": digest(a), "reverse_sha256": digest(b),
                                          "scored_tokens": int(finite.sum())}
                deltas = {}
                for name in PAIRWISE:
                    a, b = paired[name], repeated[name]
                    exact = bool(np.array_equal(a, b, equal_nan=True))
                    finite = np.isfinite(a) & np.isfinite(b)
                    error = np.abs(a[finite] - b[finite])
                    max_pair = float(error.max(initial=0.0))
                    max_abs = max(max_abs, max_pair)
                    all_exact &= exact
                    deltas[name] = {"exact": exact, "max_abs_delta": max_pair,
                                    "first_sha256": digest(a), "reverse_sha256": digest(b),
                                    "scored_tokens": int(finite.sum())}
                traces[index] = {"conditions": records, "pairwise": deltas,
                                 "passed_at_1e-6": max_abs <= 1e-6,
                                 "bitwise_exact": all_exact, "max_abs_delta": max_abs}
    example_records = []
    passed = True
    for index, body, bucket in selected:
        item = traces[index]
        passed &= item["passed_at_1e-6"]
        example_records.append({"row_key": rows[index]["row_key"],
                                "original_index": rows[index]["original_index"],
                                "script_bucket": bucket, "body_tokens": len(body), **item})
    return {"check": "singleton forward then reverse article-order repeat",
            "scoring_batch_size": 1, "precision": "bfloat16", "attention_backend": "sdpa",
            "selected_examples": example_records, "selected_script_buckets": sorted({x[2] for x in selected}),
            "body_token_lengths": [len(x[1]) for x in selected],
            "elapsed_seconds": time.monotonic() - start,
            "tolerance_abs": 1e-6, "passed": bool(passed),
            "note": "Each article-condition uses identical singleton tensor shape in both orders; per-token NLL and all aligned pairwise deltas are hashed and compared."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/condition_probe.json"))
    ap.add_argument("--splits", nargs="+", choices=("train", "dev"), default=["train"])
    ap.add_argument("--limit", type=int, help="uniform deterministic pilot size; train only")
    ap.add_argument("--shard-rows", type=int, help="rows per atomic NPZ shard; model scoring remains singleton")
    ap.add_argument("--verify-singleton-repeat", action="store_true")
    ap.add_argument("--skip-model-load", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["model_name"] != MODEL_NAME or cfg["revision"] != MODEL_REVISION:
        raise ValueError("model name/revision are fixed by the condition probe contract")
    if cfg.get("precision") != "bfloat16" or cfg.get("attention_backend") != "sdpa":
        raise ValueError("condition probe requires BF16 and SDPA")
    if cfg.get("scoring_batch_size") != 1 or int(cfg.get("shard_rows", 0)) < 1:
        raise ValueError("production condition-probe scoring batch is fixed at one; shard_rows is separate")
    out_root = (ROOT / cfg["output_dir"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runner_lock = RunLock(out_root / ".runner.lock")
    runner_lock.__enter__()
    experiment = {"schema_version": "condition_probe_experiment_v1",
                  "config_sha256": sha256_file(cfg_path), "model": MODEL_NAME,
                  "model_revision": MODEL_REVISION,
                  "splits_requested": args.splits, "pilot_limit": args.limit,
                  "scoring_batch_size": 1,
                  "shard_rows": args.shard_rows or cfg["shard_rows"],
                  "templates": TEMPLATES}
    experiment_path = out_root / "experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != experiment:
        raise ValueError("output root belongs to a different condition-probe experiment")
    if not experiment_path.exists():
        if any(p.name != ".runner.lock" for p in out_root.iterdir()):
            raise ValueError("output root contains unmanifested artifacts; refusing overwrite")
        atomic_json(experiment_path, experiment)

    data = {}
    frozen_paths = {}
    for split in args.splits:
        path = (ROOT / cfg[f"{split}_path"]).resolve()
        frozen_paths[split] = validate_split_path(ROOT, split, path)
        rows = read_rows(path, split)
        data[split] = select_train_pilot(rows, args.limit, split)
        print(json.dumps({"event": "selection", "split": split,
                          "selected": len(data[split]), "full_split_rows": len(rows),
                          "strategy": "uniform_train_indices" if args.limit and split == "train" else "full_frozen_split"}), flush=True)

    if args.skip_model_load:
        # CLI path validation can be checked without importing/loading model weights.
        runner_lock.__exit__(None, None, None)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen condition probe requires CUDA")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                              local_files_only=True)
    prefix_ids, _ = encode_contract(tokenizer)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True,
        attn_implementation="sdpa")
    model.eval()
    print(json.dumps({"event": "model_loaded", "seconds": time.monotonic() - t0,
                      "dtype": str(next(model.parameters()).dtype),
                      "device": str(next(model.parameters()).device)}), flush=True)

    # Pilot repeat selection is made from train rows only; labels never enter scoring.
    if args.verify_singleton_repeat:
        if "train" not in data:
            raise ValueError("singleton repeat verification requires train rows")
        check = verify_singleton_repeat(model, tokenizer, data["train"], prefix_ids)
        atomic_json(out_root / "singleton_repeat_check.json", check)
        print(json.dumps({"event": "singleton_repeat_check", **check}, ensure_ascii=False), flush=True)
        if not check["passed"]:
            raise RuntimeError("singleton score repeated differently across article order")

    for split in args.splits:
        split_out = out_root / split
        manifest = make_manifest(split=split, input_path=frozen_paths[split], tokenizer=tokenizer,
                                 prefix_ids=prefix_ids, config_path=cfg_path)
        # Selection is explicitly bound into run metadata so a pilot cannot be
        # mistaken for a full split. Full runs retain all rows in source order.
        manifest["selection"] = {"selected_rows": len(data[split]),
                                 "row_keys_sha256": hashlib.sha256("\n".join(
                                     r["row_key"] for r in data[split]).encode()).hexdigest(),
                                 "strategy": "uniform_train_indices" if args.limit and split == "train" else "full_split"}
        manifest["scoring_batch_size"] = 1
        manifest["scoring_policy"] = "single_article_forward_per_condition_no_mixed_padding"
        manifest["shard_rows"] = args.shard_rows or cfg["shard_rows"]
        result = extract_split(rows=data[split], split=split, out=split_out,
                               tokenizer=tokenizer, model=model, manifest=manifest,
                               prefix_ids=prefix_ids,
                               shard_rows=args.shard_rows or cfg["shard_rows"],
                               scoring_batch_size=cfg["scoring_batch_size"])
        print(json.dumps({"event": "split_complete", **result}, ensure_ascii=False), flush=True)
    runner_lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
