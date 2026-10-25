"""Leakage-aware CPU baselines for frozen condition-probe measurements."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, f1_score, log_loss, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .ood import connected_components

SEED = 20261019
ROLE_FRACTIONS = {"R": 0.2, "F": 0.6, "S": 0.2}
MIN_REFERENCE_CELL = 200
BOOTSTRAP_SEED = 20261020
WINDOW_COUNT = 28
COMPARISONS = ("A", "B", "C_neutral", "C", "D_global", "D")
SUMMARY_FIELDS = ("scored_token_count", "mean", "std", "q10", "q25", "median", "q75", "q90")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_key(split: str, index: int, text: str) -> str:
    return hashlib.sha256(f"{split}\0{index}\0{text}".encode("utf-8")).hexdigest()


def _choose_group_subset(components: list[list[int]], candidates: list[int], rows: list[dict[str, Any]],
                         target_rows: int, rng: np.random.Generator) -> set[int]:
    """One seeded whole-group pass, with deterministic class-coverage completion."""
    if not candidates:
        raise ValueError("no connected components available for requested role")
    order = [int(x) for x in rng.permutation(np.asarray(candidates, dtype=np.int64))]
    chosen: set[int] = set()
    n = 0
    for gid in order:
        if n >= target_rows:
            break
        chosen.add(gid)
        n += len(components[gid])
    # Add the first seeded group carrying any missing class; do not change seed or retry.
    for label in (0, 1):
        while not any(int(rows[i]["label"]) == label for gid in chosen for i in components[gid]):
            next_gid = next((gid for gid in order if gid not in chosen and
                             any(int(rows[i]["label"]) == label for i in components[gid])), None)
            if next_gid is None:
                raise ValueError(f"cannot provide label {label} in role")
            chosen.add(next_gid)
    return chosen


def make_role_split(rows: list[dict[str, Any]], *, seed: int = SEED,
                    fractions: dict[str, float] | None = None) -> dict[str, Any]:
    """Assign a train-only pool to human-reference R and classifier F/S roles.

    Components may only appear in one role. R components can contain either
    label; only their human rows are used to fit the reference. F/S are
    guaranteed to contain both labels.
    """
    fractions = fractions or ROLE_FRACTIONS
    if set(fractions) != {"R", "F", "S"} or not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("role fractions must define R/F/S and sum to one")
    if any(int(r["label"]) not in (0, 1) for r in rows):
        raise ValueError("labels must be binary")
    components, row_groups = connected_components(rows)
    rng = np.random.default_rng(seed)
    all_groups = list(range(len(components)))
    r_groups = _choose_group_subset(components, all_groups, rows,
                                    max(1, round(len(rows) * fractions["R"])), rng)
    remaining = [gid for gid in range(len(components)) if gid not in r_groups]
    remaining_rows = sum(len(components[gid]) for gid in remaining)
    # F:S is 3:1 after R has been removed.
    s_groups = _choose_group_subset(components, remaining, rows,
                                    max(1, round(remaining_rows * fractions["S"] /
                                                 (fractions["F"] + fractions["S"]))), rng)
    f_groups = set(remaining) - s_groups
    roles = {"R": r_groups, "F": f_groups, "S": s_groups}
    row_roles: list[str] = [""] * len(rows)
    for role, group_ids in roles.items():
        for gid in group_ids:
            for ix in components[gid]:
                row_roles[ix] = role
    if any(not role for role in row_roles):
        raise AssertionError("a row did not receive a role")
    for role in ("F", "S"):
        labels = {int(rows[i]["label"]) for i, item_role in enumerate(row_roles) if item_role == role}
        if labels != {0, 1}:
            raise ValueError(f"{role} role is missing a label: {sorted(labels)}")
    if not any(int(rows[i]["label"]) == 0 and role == "R" for i, role in enumerate(row_roles)):
        raise ValueError("R contains no human rows for reference fitting")
    role_sets = {role: set(gids) for role, gids in roles.items()}
    if any(role_sets[a] & role_sets[b] for a, b in (("R", "F"), ("R", "S"), ("F", "S"))):
        raise AssertionError("component crosses roles")
    counts = {role: sum(item == role for item in row_roles) for role in roles}
    return {"components": components, "row_groups": row_groups, "row_roles": row_roles,
            "role_group_ids": roles, "role_row_counts": counts, "seed": seed,
            "requested_fractions": dict(fractions), "cross_role_components": 0}


def manifest_rows(rows: list[dict[str, Any]], split: str, *, row_roles: list[str] | None = None,
                  group_ids: list[int] | None = None, role: str = "DEV") -> list[dict[str, Any]]:
    """Create text-free join records using original split-local indices."""
    result = []
    for i, row in enumerate(rows):
        original_split = str(row.get("_original_split", split))
        index = int(row.get("_original_index", i))
        text = str(row["text"])
        result.append({"row_key": row_key(original_split, index, text), "text_sha256": text_sha256(text),
                       "original_split": original_split, "original_index": index,
                       "group_id": int(group_ids[i]) if group_ids is not None else None,
                       "role": row_roles[i] if row_roles is not None else role,
                       "label": int(row["label"])})
    return result


def text_background_features(text: str) -> tuple[dict[str, float], str, str]:
    """Body-only B features plus coarse, text-derived D reference keys."""
    chars = list(text)
    n = max(1, len(chars))
    words = text.split()
    letter = digit = punct = whitespace = 0
    scripts: Counter[str] = Counter()
    for ch in chars:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            letter += 1
            name = unicodedata.name(ch, "")
            if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
                scripts["cjk"] += 1
            elif "LATIN" in name:
                scripts["latin"] += 1
            elif "CYRILLIC" in name:
                scripts["cyrillic"] += 1
            elif "ARABIC" in name:
                scripts["arabic"] += 1
            else:
                scripts["other"] += 1
        elif cat.startswith("N"):
            digit += 1
        elif cat.startswith("P") or cat.startswith("S"):
            punct += 1
        if ch.isspace():
            whitespace += 1
    dominant = max(sorted(scripts), key=lambda k: scripts[k]) if scripts else "none"
    length = len(words)
    length_bucket = ("0-63" if length < 64 else "64-127" if length < 128 else
                     "128-255" if length < 256 else "256-511" if length < 512 else "512+")
    features = {
        "char_count": float(len(chars)), "log_char_count": math.log1p(len(chars)),
        "whitespace_token_count": float(length), "log_whitespace_token_count": math.log1p(length),
        "whitespace_count": float(whitespace), "letter_ratio": letter / n,
        "digit_ratio": digit / n, "punctuation_symbol_ratio": punct / n,
        "whitespace_ratio": whitespace / n,
        "latin_letter_ratio": scripts["latin"] / n, "cjk_letter_ratio": scripts["cjk"] / n,
        "cyrillic_letter_ratio": scripts["cyrillic"] / n, "arabic_letter_ratio": scripts["arabic"] / n,
        "other_letter_ratio": scripts["other"] / n,
    }
    return features, length_bucket, dominant


def body_token_length_bucket(count: int) -> str:
    return ("0-63" if count < 64 else "64-127" if count < 128 else
            "128-255" if count < 256 else "256-447" if count < 448 else "448")


def reference_bucket(text: str, body_token_count: int) -> tuple[str, str]:
    _, _, script = text_background_features(text)
    return body_token_length_bucket(int(body_token_count)), script


def _checked_count(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"invalid observation count for {field}: {value!r}")
    return float(value)


def _flatten_summary(prefix: str, summary: dict[str, Any]) -> dict[str, float]:
    """Flatten to the declared schema, zero-filling unavailable summary values.

    A zero scored-token count explicitly marks empty summaries. Probe JSON encodes
    their scalar statistics as null, so conditional key emission would otherwise
    make a single short row change every downstream feature matrix's schema.
    """
    if "scored_token_count" not in summary:
        raise ValueError(f"summary lacks scored_token_count for {prefix}")
    scored_count = _checked_count(summary["scored_token_count"], f"{prefix}_scored_token_count")
    out: dict[str, float] = {f"{prefix}_scored_token_count": scored_count}
    for key in SUMMARY_FIELDS:
        if key == "scored_token_count":
            continue
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[f"{prefix}_{key}"] = float(value)
        elif scored_count == 0:
            out[f"{prefix}_{key}"] = 0.0
        else:
            raise ValueError(f"missing/non-finite {key} with scored_token_count={scored_count:g} for {prefix}")
    windows = summary.get("window16_means", []) or []
    for ix in range(WINDOW_COUNT):
        value = windows[ix] if ix < len(windows) else None
        mean = value.get("mean") if isinstance(value, dict) else value
        if isinstance(value, dict):
            window_count = _checked_count(value.get("count", 0), f"{prefix}_window16_count_{ix:03d}")
        else:
            window_count = 1.0 if isinstance(mean, (int, float)) and math.isfinite(float(mean)) else 0.0
        if isinstance(mean, (int, float)) and math.isfinite(float(mean)):
            out[f"{prefix}_window16_{ix:03d}"] = float(mean)
        elif window_count == 0:
            out[f"{prefix}_window16_{ix:03d}"] = 0.0
        else:
            raise ValueError(f"missing/non-finite window mean with count={window_count:g} for {prefix} window {ix}")
    return out


def probe_features(probe_row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Return A raw features and C condition-response deltas from probe row."""
    conditions = probe_row.get("conditions") or {}
    if not all(name in conditions for name in ("raw", "neutral", "human_hypothesis", "machine_hypothesis")):
        raise ValueError("probe row lacks one of the four frozen conditions")
    a = _flatten_summary("raw", conditions["raw"])
    # A is the established aggregate raw-token statistic set. Local windows
    # enter only with the explicit condition-response deltas in C.
    a = {key: value for key, value in a.items() if not key.startswith("raw_window16_")}
    deltas = probe_row.get("pairwise_surprise_deltas") or {}
    c: dict[str, float] = {}
    required = ("neutral_minus_raw", "human_minus_neutral", "machine_minus_neutral", "machine_minus_human")
    for name in required:
        if name not in deltas:
            raise ValueError(f"probe row lacks pairwise delta {name}")
        c.update(_flatten_summary(name, deltas[name]))
    # A fixed 28-window schema avoids losing later windows merely because a
    # short article has no corresponding window. Counts remain explicit.
    for name in required:
        summary = deltas[name]
        for ix in range(WINDOW_COUNT):
            windows = summary.get("window16_means", []) or []
            if ix < len(windows) and isinstance(windows[ix], dict):
                count = _checked_count(windows[ix].get("count", 0), f"{name}_window16_count_{ix:03d}")
            elif ix < len(windows) and isinstance(windows[ix], (int, float)) and math.isfinite(float(windows[ix])):
                count = 1.0
            else:
                count = 0.0
            c[f"{name}_window16_count_{ix:03d}"] = count
    return a, c


def align_probe_rows(source_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Strict exact-once alignment and audit checks for a frozen source split."""
    by_key: dict[str, dict[str, Any]] = {}
    for row in probe_rows:
        key = str(row.get("row_key", ""))
        if not key or key in by_key:
            raise ValueError("probe row_key missing or duplicated")
        by_key[key] = row
    aligned = []
    expected_keys = set()
    for i, row in enumerate(source_rows):
        text = str(row["text"])
        key = row_key(split, i, text)
        expected_keys.add(key)
        feat = by_key.get(key)
        if feat is None:
            raise ValueError(f"probe missing {split} row {i}")
        if feat.get("text_sha256") != text_sha256(text):
            raise ValueError(f"probe text hash mismatch at {split} row {i}")
        if feat.get("original_index") != i:
            raise ValueError(f"probe original index mismatch at {split} row {i}")
        if int(feat.get("label", row["label"])) != int(row["label"]):
            raise ValueError(f"probe label mismatch at {split} row {i}")
        aligned.append(feat)
    if set(by_key) != expected_keys:
        raise ValueError("probe contains rows outside the frozen split or misses expected rows")
    return aligned


def fit_human_reference(features: np.ndarray, buckets: list[tuple[str, str]], *,
                         min_cell_rows: int = MIN_REFERENCE_CELL) -> dict[str, Any]:
    """Fit robust z and empirical percentile parameters using human R only."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(buckets) or not np.isfinite(x).all() or not len(x):
        raise ValueError("reference matrix must be finite, nonempty, and aligned")
    def stats(values: np.ndarray) -> dict[str, Any]:
        med = np.median(values, axis=0)
        mad = np.median(np.abs(values - med), axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        scale = 1.4826 * mad
        global_mad_scale = 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)
        global_iqr_scale = (np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)) / 1.349
        global_fallback = np.where(global_mad_scale > 1e-12, global_mad_scale,
                                   np.where(global_iqr_scale > 1e-12, global_iqr_scale, 1.0))
        scale = np.where(scale > 1e-12, scale, global_fallback)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return {"median": med, "mad": mad, "scale": scale,
                "sorted": np.sort(values, axis=0), "n": int(len(values))}
    global_stats = stats(x)
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in sorted(set(buckets)):
        ix = np.asarray([b == cell for b in buckets])
        if int(ix.sum()) >= min_cell_rows:
            cells[cell] = stats(x[ix])
    return {"global": global_stats, "cells": cells, "min_cell_rows": min_cell_rows,
            "feature_count": int(x.shape[1]), "reference_rows": int(len(x))}


def apply_human_reference(features: np.ndarray, buckets: list[tuple[str, str]],
                          reference: dict[str, Any], *, use_global: bool = False) -> np.ndarray:
    """Append per-feature robust z and empirical midrank percentile."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != reference["feature_count"] or len(buckets) != len(x):
        raise ValueError("reference transform shape mismatch")
    result = []
    for row, bucket in zip(x, buckets):
        stats = reference["global"] if use_global else reference["cells"].get(tuple(bucket), reference["global"])
        z = (row - stats["median"]) / stats["scale"]
        sorted_values = stats["sorted"]
        percentile = np.asarray([(np.searchsorted(sorted_values[:, j], row[j], side="left") +
                                  np.searchsorted(sorted_values[:, j], row[j], side="right")) /
                                 (2.0 * len(sorted_values)) for j in range(x.shape[1])])
        result.append(np.concatenate([row, z, percentile]))
    return np.asarray(result, dtype=np.float64)


def build_feature_matrices(rows: list[dict[str, Any]], aligned_probe: list[dict[str, Any]],
                           *, min_cell_rows: int = MIN_REFERENCE_CELL) -> tuple[dict[str, np.ndarray], list[str], list[tuple[str, str]], dict[str, Any]]:
    """Build shared-row A/B/C/D matrices; reference fit must be called on R only."""
    if len(rows) != len(aligned_probe):
        raise ValueError("source and probe row counts differ")
    a_rows, b_rows, c_rows, buckets = [], [], [], []
    for source, probe in zip(rows, aligned_probe):
        a, c = probe_features(probe)
        b, _, _ = text_background_features(str(source["text"]))
        token_count = int(probe.get("body_token_count", len(str(source["text"]).split())))
        bucket = reference_bucket(str(source["text"]), token_count)
        a_rows.append(a)
        b_rows.append(b)
        c_rows.append(c)
        buckets.append(bucket)
    if not a_rows:
        raise ValueError("cannot build feature matrices from zero rows")
    a_names, b_names, c_names = sorted(a_rows[0]), sorted(b_rows[0]), sorted(c_rows[0])
    for i, (a_row, b_row, c_row) in enumerate(zip(a_rows, b_rows, c_rows)):
        if sorted(a_row) != a_names or sorted(b_row) != b_names or sorted(c_row) != c_names:
            raise ValueError(f"feature schema drift at row {i}")
    if not a_names or not c_names:
        raise ValueError("empty probe feature schema")
    a = np.asarray([[x[k] for k in a_names] for x in a_rows], dtype=np.float64)
    b = np.asarray([[x[k] for k in b_names] for x in b_rows], dtype=np.float64)
    c = np.asarray([[x[k] for k in c_names] for x in c_rows], dtype=np.float64)
    neutral_names = sorted(k for k in c_names if k.startswith("neutral_minus_raw_"))
    neutral = np.asarray([[x[k] for k in neutral_names] for x in c_rows], dtype=np.float64)
    matrices = {"A": a, "B": np.column_stack([a, b]),
                "C_neutral": np.column_stack([a, b, neutral]),
                "C": np.column_stack([a, b, c])}
    names = {"A": a_names, "B": a_names + b_names,
             "C_neutral": a_names + b_names + neutral_names,
             "C": a_names + b_names + c_names}
    return matrices, names, buckets, {"A": a_names, "B": b_names, "C": c_names}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc


def build_feature_matrices_stream(rows: list[dict[str, Any]], probe_path: Path, split: str,
                                 *, min_cell_rows: int = MIN_REFERENCE_CELL):
    """Consume one large nested probe JSONL incrementally into compact arrays."""
    a_rows, b_rows, c_rows, buckets = [], [], [], []
    a_names = c_names = b_names = None
    probe_iter = iter_jsonl(probe_path)
    for i, (source, probe) in enumerate(zip(rows, probe_iter)):
        text = str(source["text"])
        expected_key = row_key(split, i, text)
        if probe.get("row_key") != expected_key or probe.get("original_index") != i:
            raise ValueError(f"{split} feature alignment mismatch at row {i}")
        if probe.get("text_sha256") != text_sha256(text) or int(probe.get("label", source["label"])) != int(source["label"]):
            raise ValueError(f"{split} feature content/label mismatch at row {i}")
        a, c = probe_features(probe)
        b, _, _ = text_background_features(text)
        if "body_token_count" not in probe:
            raise ValueError(f"{split} probe lacks body_token_count at row {i}")
        bucket = reference_bucket(text, int(probe["body_token_count"]))
        current_a, current_b, current_c = sorted(a), sorted(b), sorted(c)
        if a_names is None:
            a_names, b_names, c_names = current_a, current_b, current_c
        if current_a != a_names or current_b != b_names or current_c != c_names:
            raise ValueError(f"{split} feature schema drift at row {i}")
        a_rows.append([a[k] for k in a_names])
        b_rows.append([b[k] for k in b_names])
        c_rows.append([c[k] for k in c_names])
        buckets.append(bucket)
    try:
        next(probe_iter)
    except StopIteration:
        pass
    else:
        raise ValueError(f"{split} probe contains extra rows")
    if len(a_rows) != len(rows) or a_names is None:
        raise ValueError(f"{split} probe row count differs from frozen source")
    a = np.asarray(a_rows, dtype=np.float32)
    b = np.asarray(b_rows, dtype=np.float32)
    c = np.asarray(c_rows, dtype=np.float32)
    neutral_names = sorted(k for k in c_names if k.startswith("neutral_minus_raw_"))
    neutral = c[:, [c_names.index(k) for k in neutral_names]]
    matrices = {"A": a, "B": np.column_stack([a, b]),
                "C_neutral": np.column_stack([a, b, neutral]),
                "C": np.column_stack([a, b, c])}
    names = {"A": a_names, "B": a_names + b_names,
             "C_neutral": a_names + b_names + neutral_names,
             "C": a_names + b_names + c_names}
    return matrices, names, buckets, {"A": a_names, "B": b_names, "C": c_names}


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    tmp.replace(path)


def fit_bounded_comparisons(train_matrices: dict[str, np.ndarray], dev_matrices: dict[str, np.ndarray],
                            y: np.ndarray, roles: list[str], dev_y: np.ndarray, *,
                            train_short: np.ndarray | None = None, dev_short: np.ndarray | None = None,
                            seed: int = SEED, cpu_threads: int = 2) -> dict[str, Any]:
    """Fit fixed LR/HGB on F, select per set on S, then score untouched dev."""
    if set(train_matrices) != set(COMPARISONS) or set(dev_matrices) != set(train_matrices):
        raise ValueError("expected exactly the predeclared A/B/C_neutral/C/D_global/D comparison sets")
    y = np.asarray(y, dtype=np.int64)
    dev_y = np.asarray(dev_y, dtype=np.int64)
    train_short = np.asarray(train_short, dtype=bool) if train_short is not None else np.zeros(len(y), dtype=bool)
    dev_short = np.asarray(dev_short, dtype=bool) if dev_short is not None else np.zeros(len(dev_y), dtype=bool)
    if len(train_short) != len(y) or len(dev_short) != len(dev_y):
        raise ValueError("short-text masks are not aligned")
    fit_ix = np.flatnonzero(np.asarray(roles) == "F")
    sel_ix = np.flatnonzero(np.asarray(roles) == "S")
    if len(fit_ix) == 0 or len(sel_ix) == 0 or set(y[fit_ix]) != {0, 1} or set(y[sel_ix]) != {0, 1}:
        raise ValueError("F/S must both contain both labels")
    if set(dev_y) != {0, 1}:
        raise ValueError("dev must contain both labels")
    outcomes: dict[str, Any] = {"selection_rows": {"F": len(fit_ix), "S": len(sel_ix)},
                                "by_comparison": {}, "models": {}, "probabilities": {}}
    with threadpool_limits(limits=cpu_threads):
        for set_name in COMPARISONS:
            x = np.asarray(train_matrices[set_name], dtype=np.float64)
            xd = np.asarray(dev_matrices[set_name], dtype=np.float64)
            if x.shape[0] != len(y) or xd.shape[0] != len(dev_y) or not np.isfinite(x).all() or not np.isfinite(xd).all():
                raise ValueError(f"{set_name} feature matrix is invalid or misaligned")
            candidates = {
                "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                                          random_state=seed)),
                "HGB": HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300,
                    max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=30,
                    early_stopping=False, random_state=seed),
            }
            selection = {}
            selected_probabilities = {}
            for name, model in candidates.items():
                model.fit(x[fit_ix], y[fit_ix])
                p = model.predict_proba(x[sel_ix])[:, 1]
                selection[name] = score_probabilities(y[sel_ix], p, train_short[sel_ix])
                selected_probabilities[name] = p
            winner = sorted(candidates, key=lambda name: (-selection[name]["combined_score"], name))[0]
            chosen = candidates[winner]
            # Refit once on F after choice; S remains a selection-only partition.
            chosen.fit(x[fit_ix], y[fit_ix])
            pd = chosen.predict_proba(xd)[:, 1]
            outcomes["by_comparison"][set_name] = {
                "selection_scores": selection, "selected_classifier": winner,
                "selected_s_combined_score": selection[winner]["combined_score"],
                "selection_calibration": {name: calibration_rows(y[sel_ix], selected_probabilities[name])
                                          for name in candidates},
                "dev_metrics": score_probabilities(dev_y, pd, dev_short),
            }
            outcomes["models"][set_name] = chosen
            outcomes["probabilities"][set_name] = {"S": selected_probabilities[winner], "DEV": pd,
                                                   "S_candidates": selected_probabilities}
    return outcomes


def score_probabilities(y: np.ndarray, probability: np.ndarray,
                        short_mask: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    pred = (p >= 0.5).astype(np.int64)
    macro = float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
    short_mask = np.asarray(short_mask, dtype=bool) if short_mask is not None else np.zeros(len(y), dtype=bool)
    if len(short_mask) != len(y):
        raise ValueError("short-text mask length differs from labels")
    if short_mask.any():
        short_macro = float(f1_score(y[short_mask], pred[short_mask], labels=[0, 1],
                                     average="macro", zero_division=0))
        short_fallback = None
    else:
        short_macro = macro
        short_fallback = "no short rows; use all-row macro F1"
    fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)
    return {"n": int(len(y)), "short_n": int(short_mask.sum()), "macro_f1": macro,
            "short_macro_f1": short_macro, "combined_score": float(.7 * macro + .3 * short_macro),
            "short_fallback": short_fallback, "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "human_fpr": float(np.mean(pred[y == 0])) if np.any(y == 0) else None,
            "machine_recall": float(np.mean(pred[y == 1])) if np.any(y == 1) else None,
            "roc": [{"fpr": float(a), "tpr": float(b),
                     "threshold": float(c) if math.isfinite(float(c)) else None}
                    for a, b, c in zip(fpr, tpr, thresholds)]}


def calibration_rows(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for i in range(bins):
        mask = ((p >= edges[i]) & (p < edges[i + 1])) if i < bins - 1 else ((p >= edges[i]) & (p <= edges[i + 1]))
        if np.any(mask):
            result.append({"bin": i, "n": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                           "fraction_machine": float(y[mask].mean())})
    return result


def paired_group_bootstrap(y: np.ndarray, probabilities: dict[str, np.ndarray], groups: np.ndarray,
                           *, replicates: int = 200, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Paired bootstrap by connected component, returning reproducible metric draws."""
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("group bootstrap needs at least two components")
    if any(len(p) != len(y) for p in probabilities.values()):
        raise ValueError("bootstrap probability vectors are not aligned")
    member_ix = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in probabilities}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([member_ix[g] for g in sampled])
        if len(np.unique(y[ix])) != 2:
            continue
        for name, p in probabilities.items():
            metrics = score_probabilities(y[ix], np.asarray(p)[ix])
            draws[name].append({k: metrics[k] for k in ("macro_f1", "roc_auc", "brier", "human_fpr", "machine_recall")})
    summary = {}
    for name, values in draws.items():
        summary[name] = {}
        for metric in ("macro_f1", "roc_auc", "brier", "human_fpr", "machine_recall"):
            vals = [float(row[metric]) for row in values if row[metric] is not None]
            summary[name][metric] = {"low_2_5": float(np.quantile(vals, .025)),
                                     "median": float(np.quantile(vals, .5)),
                                     "high_97_5": float(np.quantile(vals, .975))} if vals else None
    return {"seed": seed, "requested_replicates": replicates,
            "valid_replicates": min((len(v) for v in draws.values()), default=0),
            "unit": "connected_component", "paired": True, "summary": summary, "draws": draws}
