"""Strict source and generator-family holdout construction.

The OOD pool is deliberately limited to the already-finalized original train
and dev files. The locked test file is never opened by this module.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MIN_TRAIN_ROWS = 20_000
MIN_CLASS_ROWS = 2_000
GENERATOR_TARGET = "davinci"
DOMAIN_TARGET = "wikihow"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def load_development_pool(train_path: str | Path, dev_path: str | Path) -> list[dict[str, Any]]:
    """Read only the original train/dev final dedup records and tag provenance."""
    rows: list[dict[str, Any]] = []
    for split, path in (("train", train_path), ("dev", dev_path)):
        with Path(path).open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {split} at line {line_no}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {split} at line {line_no}")
                record = dict(row)
                record["_original_split"] = split
                rows.append(record)
    return rows


def _families(row: dict[str, Any]) -> set[str]:
    values = [row.get("generator_family", "unknown"), *(row.get("duplicate_generator_families") or [])]
    return {str(v).strip().casefold() for v in values if str(v).strip()}


def _sources(row: dict[str, Any]) -> set[str]:
    values = [row.get("source", "unknown"), *(row.get("duplicate_sources") or [])]
    return {str(v).strip() for v in values if str(v).strip()}


def _article_ids(row: dict[str, Any]) -> set[str]:
    values = [row.get("article_id"), *(row.get("duplicate_article_ids") or [])]
    return {str(v) for v in values if v is not None and str(v)}


def _body_hashes(row: dict[str, Any]) -> set[str]:
    values = [row.get("body_hash"), *(row.get("alias_body_hashes") or [])]
    return {str(v) for v in values if v}


def connected_components(rows: list[dict[str, Any]]) -> tuple[list[list[int]], list[int]]:
    """Connect rows by any article alias or exact body/alias hash."""
    uf = _UnionFind(len(rows))
    first: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        keys = [("article", key) for key in _article_ids(row)]
        keys.extend(("body", key) for key in _body_hashes(row))
        for key in keys:
            if key in first:
                uf.union(i, first[key])
            else:
                first[key] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    components = list(groups.values())
    row_group = [0] * len(rows)
    for group_id, members in enumerate(components):
        for i in members:
            row_group[i] = group_id
    return components, row_group


def _class_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(int(row["label"])) for row in rows)
    return {"0": counts["0"], "1": counts["1"]}


def _family_set(rows: Iterable[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        result.update(_families(row))
    return result


def _group_families(rows: list[dict[str, Any]], members: list[int]) -> set[str]:
    return _family_set(rows[i] for i in members)


def _group_sources(rows: list[dict[str, Any]], members: list[int]) -> set[str]:
    result: set[str] = set()
    for i in members:
        result.update(_sources(rows[i]))
    return result


def _candidate_stats(rows: list[dict[str, Any]], components: list[list[int]]) -> dict[str, Any]:
    families = sorted({fam for row in rows for fam in _families(row)} - {"human", "unknown"})
    candidates: dict[str, Any] = {}
    for target in families:
        held_groups = [g for g in components if target in _group_families(rows, g)]
        held_ids = {i for group in held_groups for i in group}
        eval_rows = [rows[i] for i in held_ids
                     if int(rows[i]["label"]) == 0 or target in _families(rows[i])]
        train_rows = [row for i, row in enumerate(rows)
                      if i not in held_ids and row["_original_split"] == "train"]
        dev_rows = [row for i, row in enumerate(rows)
                    if i not in held_ids and row["_original_split"] == "dev"]
        class_counts = _class_counts(eval_rows)
        train_class_counts = _class_counts(train_rows)
        candidates[target] = {
            "held_component_count": len(held_groups), "held_row_count": len(held_ids),
            "eval_row_count": len(eval_rows), "eval_class_counts": class_counts,
            "retained_train_count": len(train_rows), "retained_dev_count": len(dev_rows),
            "retained_train_class_counts": train_class_counts,
            "retained_dev_class_counts": _class_counts(dev_rows),
            "eligible": (len(train_rows) >= MIN_TRAIN_ROWS
                         and train_class_counts["0"] >= MIN_CLASS_ROWS
                         and train_class_counts["1"] >= MIN_CLASS_ROWS
                         and class_counts["0"] >= MIN_CLASS_ROWS
                         and class_counts["1"] >= MIN_CLASS_ROWS),
        }
    return candidates


def choose_generator_target(candidates: dict[str, Any]) -> str:
    eligible = [name for name, stats in candidates.items() if stats["eligible"]]
    if not eligible:
        raise ValueError("no generator family meets fixed OOD eligibility thresholds")
    # Fixed, score-blind rule: maximize target evaluation rows, then family name.
    return sorted(eligible, key=lambda name: (-candidates[name]["eval_row_count"], name))[0]


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "class_counts": _class_counts(rows),
        "primary_source_counts": dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items())),
        "primary_family_counts": dict(sorted(Counter(str(row.get("generator_family", "unknown")) for row in rows).items())),
        "source_counts": dict(sorted(Counter(
            source for row in rows for source in _sources(row)
        ).items())),
        "family_counts": dict(sorted(Counter(
            family for row in rows for family in _families(row)
        ).items())),
    }


def _assert_valid_protocol(name: str, rows: list[dict[str, Any]], components: list[list[int]],
                           splits: dict[str, list[int]], held_groups: set[int],
                           eval_target: str, target_type: str) -> None:
    train, dev, evaluation = (set(splits[key]) for key in ("train", "dev", "eval"))
    if train & dev or train & evaluation or dev & evaluation:
        raise AssertionError(f"{name}: row overlap across splits")
    split_sets = {key: set(indexes) for key, indexes in splits.items()}
    for group_id, members in enumerate(components):
        split_names = {key for key, indexes in split_sets.items() if any(i in indexes for i in members)}
        if len(split_names) > 1:
            raise AssertionError(f"{name}: connected component split across {sorted(split_names)}")
    for idx in train | dev:
        if rows[idx]["_original_split"] not in {"train", "dev"}:
            raise AssertionError(f"{name}: row did not originate in original train/dev")
    if not train or not dev or not evaluation:
        raise ValueError(f"{name}: train, dev, and eval must all be nonempty")
    for split_name, indexes in (("train", train), ("dev", dev), ("eval", evaluation)):
        counts = _class_counts(rows[i] for i in indexes)
        if not all(counts.values()):
            raise ValueError(f"{name}: {split_name} is missing one of the two labels: {counts}")
    if target_type == "family":
        for i in train | dev:
            if eval_target in _families(rows[i]):
                raise AssertionError(f"{name}: target family appears in {rows[i]['_original_split']}")
        for i in evaluation:
            row = rows[i]
            if int(row["label"]) == 1 and eval_target not in _families(row):
                raise AssertionError(f"{name}: eval contains a non-target machine family")
    else:
        for i in train | dev:
            if eval_target in _sources(rows[i]):
                raise AssertionError(f"{name}: target domain appears in train/dev")
        for i in evaluation:
            if eval_target not in _sources(rows[i]):
                raise AssertionError(f"{name}: eval contains a non-target domain row")


def make_family_splits(rows: list[dict[str, Any]], components: list[list[int]], target: str) -> tuple[dict[str, list[int]], set[int]]:
    held_groups = {gid for gid, group in enumerate(components)
                   if target.casefold() in _group_families(rows, group)}
    held_rows = {i for gid in held_groups for i in components[gid]}
    splits = {
        "train": [i for i, row in enumerate(rows) if i not in held_rows and row["_original_split"] == "train"],
        "dev": [i for i, row in enumerate(rows) if i not in held_rows and row["_original_split"] == "dev"],
        "eval": [i for i in sorted(held_rows)
                 if int(rows[i]["label"]) == 0 or target.casefold() in _families(rows[i])],
    }
    return splits, held_groups


def make_domain_splits(rows: list[dict[str, Any]], components: list[list[int]], target: str) -> tuple[dict[str, list[int]], set[int]]:
    held_groups = {gid for gid, group in enumerate(components)
                   if target in _group_sources(rows, group)}
    held_rows = {i for gid in held_groups for i in components[gid]}
    splits = {
        "train": [i for i, row in enumerate(rows) if i not in held_rows and row["_original_split"] == "train"],
        "dev": [i for i, row in enumerate(rows) if i not in held_rows and row["_original_split"] == "dev"],
        "eval": [i for i in sorted(held_rows) if target in _sources(rows[i])],
    }
    return splits, held_groups


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _row_id(row: dict[str, Any]) -> str:
    payload = f"{row.get('_original_split')}:{row.get('line')}:{row.get('label')}:{row.get('body_hash')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _write_protocol(out_dir: Path, name: str, rows: list[dict[str, Any]],
                    split_indexes: dict[str, list[int]], held_groups: set[int],
                    row_group: list[int], target: str, target_type: str,
                    source_files: dict[str, str], directory_name: str) -> dict[str, Any]:
    protocol_dir = out_dir / directory_name
    protocol_dir.mkdir(parents=True, exist_ok=True)
    split_info: dict[str, Any] = {}
    for split_name, indexes in split_indexes.items():
        data_path = protocol_dir / f"{split_name}.jsonl"
        manifest_path = protocol_dir / f"{split_name}.manifest.jsonl"
        with data_path.open("w", encoding="utf-8") as data, manifest_path.open("w", encoding="utf-8") as manifest:
            for i in indexes:
                row = {k: v for k, v in rows[i].items() if not k.startswith("_")}
                data.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest.write(json.dumps({
                    "example_id": _row_id(rows[i]), "original_split": rows[i]["_original_split"],
                    "label": int(rows[i]["label"]), "body_hash": rows[i].get("body_hash"),
                    "article_ids": sorted(_article_ids(rows[i])), "sources": sorted(_sources(rows[i])),
                    "generator_families": sorted(_families(rows[i])),
                    "component_id": row_group[i],
                }, ensure_ascii=False) + "\n")
        split_info[split_name] = {
            "count": len(indexes), "sha256": _sha256(data_path),
            "manifest_sha256": _sha256(manifest_path),
            "manifest_path": str(manifest_path.relative_to(out_dir)),
            **_distribution([rows[i] for i in indexes]),
        }
    metadata = {
        "protocol": name, "directory": directory_name, "target_type": target_type, "target": target,
        "selection_is_score_blind": True, "original_sources": source_files,
        "held_component_count": len(held_groups), "splits": split_info,
        "leakage_assertions": {
            "train_dev_eval_row_disjoint": True,
            "connected_components_are_partitioned": True,
            "locked_test_read": False,
            "pool_source": ["original train.jsonl", "original dev.jsonl"],
            "group_lineage_fields": ["article_id", "duplicate_article_ids", "body_hash", "alias_body_hashes"],
            "family_lineage_fields": ["generator_family", "duplicate_generator_families"],
            "domain_lineage_fields": ["source", "duplicate_sources"],
            "target_excluded_from_train_dev": True,
            "eval_contains_only_target_group_rows": True,
        },
    }
    (protocol_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def prepare_ood(train_path: str | Path, dev_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build fixed davinci generator and wikihow domain OOD protocols."""
    train_path, dev_path, output_dir = Path(train_path), Path(dev_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"OOD output directory is nonempty; refusing to overwrite: {output_dir}")
    rows = load_development_pool(train_path, dev_path)
    components, row_group = connected_components(rows)
    source_files = {"train_sha256": _sha256(train_path), "dev_sha256": _sha256(dev_path)}
    candidates = _candidate_stats(rows, components)
    chosen_family = choose_generator_target(candidates)
    if chosen_family != GENERATOR_TARGET:
        raise ValueError(f"fixed maximum-target rule selected {chosen_family}, expected reviewed {GENERATOR_TARGET}")
    if DOMAIN_TARGET not in {source for row in rows for source in _sources(row)}:
        raise ValueError(f"fixed domain target {DOMAIN_TARGET!r} is absent")
    output_dir.mkdir(parents=True, exist_ok=True)

    family_splits, family_held_groups = make_family_splits(rows, components, chosen_family)
    _assert_valid_protocol("generator_holdout", rows, components, family_splits,
                           family_held_groups, chosen_family, "family")

    domain_splits, domain_held_groups = make_domain_splits(rows, components, DOMAIN_TARGET)
    _assert_valid_protocol("domain_holdout", rows, components, domain_splits,
                           domain_held_groups, DOMAIN_TARGET, "domain")

    protocols = {
        "generator_holdout": _write_protocol(output_dir, "generator_holdout", rows, family_splits,
                                              family_held_groups, row_group, chosen_family, "family", source_files,
                                              "generator_davinci"),
        "domain_holdout": _write_protocol(output_dir, "domain_holdout", rows, domain_splits,
                                           domain_held_groups, row_group, DOMAIN_TARGET, "domain", source_files,
                                           "domain_wikihow"),
    }
    summary = {
        "protocol_version": 1,
        "source_pool": {
            "train_path": str(train_path), "dev_path": str(dev_path),
            "train_sha256": source_files["train_sha256"], "dev_sha256": source_files["dev_sha256"],
            "row_count": len(rows), "connected_component_count": len(components),
            "locked_test_read": False,
        },
        "fixed_selection_rule": {
            "generator": "Eligible family with largest eval row count, lexical tie break; selection uses counts only.",
            "min_retained_train_rows": MIN_TRAIN_ROWS,
            "min_eval_rows_per_label": MIN_CLASS_ROWS,
            "selected_generator": chosen_family,
            "domain_priority": [DOMAIN_TARGET],
        },
        "generator_candidates": candidates,
        "protocols": protocols,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
