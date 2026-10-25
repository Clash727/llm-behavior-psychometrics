"""Teacher-forced true-token surprise statistics for a frozen causal LM.

This module deliberately never turns a single observed-token probability into
an entropy estimate: the features summarize only the observed token sequence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

FEATURE_VERSION = "true_token_surprise_v1"
DEFAULT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MAX_LENGTH = 512
LOW_PROBABILITY_CUTOFF = 0.01
LOGIT_TIME_CHUNK = 8
FEATURE_NAMES = (
    "surprise_mean", "surprise_std", "surprise_q10", "surprise_q25",
    "surprise_median", "surprise_q75", "surprise_q90",
    "low_probability_fraction", "adjacent_abs_delta_mean",
    "adjacent_abs_delta_std", "valid_token_count",
)


def summarize_surprises(surprises: Iterable[float]) -> dict[str, float | int]:
    """Summarize finite -log P(real token) values; empty/singleton safe."""
    vals = np.asarray(list(surprises), dtype=np.float64)
    if vals.ndim != 1 or not np.isfinite(vals).all():
        raise ValueError("surprises must be a finite one-dimensional sequence")
    if not len(vals):
        return {name: 0.0 for name in FEATURE_NAMES}
    delta = np.abs(np.diff(vals))
    q10, q25, median, q75, q90 = np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "surprise_mean": float(vals.mean()),
        "surprise_std": float(vals.std(ddof=0)),
        "surprise_q10": float(q10), "surprise_q25": float(q25),
        "surprise_median": float(median), "surprise_q75": float(q75),
        "surprise_q90": float(q90),
        "low_probability_fraction": float(np.mean(vals >= -math.log(LOW_PROBABILITY_CUTOFF))),
        "adjacent_abs_delta_mean": float(delta.mean()) if len(delta) else 0.0,
        "adjacent_abs_delta_std": float(delta.std(ddof=0)) if len(delta) else 0.0,
        "valid_token_count": int(len(vals)),
    }


def surprises_from_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          special_token_ids: Iterable[int] = ()) -> list[np.ndarray]:
    """Compute shifted true-token surprises from [B,T,V] logits for tests/use.

    Position zero is never scored (there is no preceding context). Padding and
    special-token targets are excluded. The caller must provide logits at
    positions t-1 to predict input_ids at positions t, as in causal LM training.
    """
    if logits.ndim != 3 or input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("expected logits [B,T,V], input_ids/mask [B,T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input sequence dimensions must match")
    if input_ids.shape[1] < 2:
        return [np.empty((0,), dtype=np.float64) for _ in range(input_ids.shape[0])]
    targets = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    specials = set(int(x) for x in special_token_ids)
    if specials:
        for token_id in specials:
            valid &= targets != token_id
    # Compute only in fp32 for numerical stability; this helper is intended for
    # small tensors. Production scoring below limits the vocabulary projection.
    log_denom = torch.logsumexp(logits[:, :-1, :].float(), dim=-1)
    true_logits = logits[:, :-1, :].float().gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    surprise = log_denom - true_logits
    return [surprise[i][valid[i]].detach().cpu().numpy().astype(np.float64) for i in range(input_ids.shape[0])]


@torch.inference_mode()
def score_batch_chunked(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        special_token_ids: Iterable[int] = (),
                        time_chunk: int = LOGIT_TIME_CHUNK) -> list[dict[str, float | int]]:
    """Score a padded batch while materializing vocab logits for <=time_chunk steps.

    Uses the base transformer and lm_head directly, so it does not retain the
    full [B,T,V] logits tensor returned by AutoModelForCausalLM.forward.
    """
    if time_chunk < 1:
        raise ValueError("time_chunk must be positive")
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError("expected a Qwen-style causal LM exposing model and lm_head")
    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                          return_dict=True)
    hidden = outputs.last_hidden_state
    batch, seq_len = input_ids.shape
    per_row: list[list[torch.Tensor]] = [[] for _ in range(batch)]
    specials = set(int(x) for x in special_token_ids)
    # Target at t is predicted by hidden state t-1. Never score token zero.
    for start in range(1, seq_len, time_chunk):
        end = min(seq_len, start + time_chunk)
        # Only produce logits for hidden positions [start-1, end-1).
        logits = model.lm_head(hidden[:, start - 1:end - 1, :])
        log_denom = torch.logsumexp(logits.float(), dim=-1)
        target = input_ids[:, start:end]
        true_logits = logits.float().gather(-1, target.unsqueeze(-1)).squeeze(-1)
        surprise = (log_denom - true_logits)
        valid = attention_mask[:, start:end].bool()
        if specials:
            for token_id in specials:
                valid &= target != token_id
        for row in range(batch):
            if valid[row].any():
                per_row[row].append(surprise[row][valid[row]].detach())
        del logits, log_denom, true_logits, surprise
    result = []
    for values in per_row:
        seq = torch.cat(values).float().cpu().numpy() if values else np.empty((0,), dtype=np.float32)
        result.append(summarize_surprises(seq))
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash tokenizer semantics while ignoring per-call padding/truncation state."""
    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    payload.pop("padding", None)
    payload.pop("truncation", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def cache_contract(*, split: str, input_hash: str, model_name: str, revision: str,
                   tokenizer_name: str, precision: str, max_length: int,
                   tokenizer_hash: str = "",
                   special_token_policy: str = "exclude_special_targets_keep_context",
                   feature_version: str = FEATURE_VERSION) -> dict[str, Any]:
    return {"split": split, "input_sha256": input_hash, "model": model_name,
            "model_revision": revision, "tokenizer": tokenizer_name,
            "tokenizer_sha256": tokenizer_hash,
            "precision": precision, "max_length": max_length,
            "truncation": "right_truncate", "feature_version": feature_version,
            "special_token_policy": special_token_policy,
            "low_probability_cutoff": LOW_PROBABILITY_CUTOFF,
            "logit_time_chunk": LOGIT_TIME_CHUNK}


def row_key(split: str, index: int, text: str) -> str:
    return hashlib.sha256(f"{split}\0{index}\0{text}".encode("utf-8")).hexdigest()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or row.get("label") not in (0, 1, "0", "1"):
                raise ValueError(f"{path}:{line_no}: expected text and binary label")
            # Keep only downstream evaluation metadata; model sees only text.
            rows.append({"text": row["text"], "label": int(row["label"]),
                         "source": row.get("source", "unknown"),
                         "row_key": row_key(Path(path).stem, len(rows), row["text"])})
    return rows


def _read_jsonl(path: Path, *, repair_partial_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    raw = path.read_bytes()
    offset = 0
    truncated_tail = False
    lines = raw.splitlines(keepends=True)
    for i, raw_line in enumerate(lines):
        next_offset = offset + len(raw_line)
        try:
            line = raw_line.decode("utf-8")
            result.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_unterminated_tail = i == len(lines) - 1 and not raw_line.endswith((b"\n", b"\r"))
            if repair_partial_tail and is_unterminated_tail:
                with path.open("r+b") as f:
                    f.truncate(offset)
                    f.flush()
                    os.fsync(f.fileno())
                truncated_tail = True
                break
            raise ValueError(f"corrupt JSONL cache at {path}, byte {offset}") from exc
        offset = next_offset
    # A syntactically valid final JSON object without newline is safe, but add
    # the delimiter before a future append so the next row cannot concatenate.
    if repair_partial_tail and raw and not truncated_tail and not raw.endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    return result


def extract_split(rows: list[dict[str, Any]], *, split: str, output_dir: str | Path,
                  tokenizer, model, contract: dict[str, Any], batch_size: int = 1,
                  limit: int | None = None, max_length: int = MAX_LENGTH) -> list[dict[str, Any]]:
    """Resumable batch extraction. Metadata contract must match exactly."""
    if split not in {"train", "dev"}:
        raise ValueError("only frozen train/dev splits are allowed")
    if batch_size < 1 or max_length != MAX_LENGTH:
        raise ValueError("batch_size must be positive and max_length must be 512")
    selected = rows[:limit] if limit is not None else rows
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / f"{split}.meta.json"
    feature_path = out / f"{split}.features.jsonl"
    progress_path = out / f"{split}.progress.jsonl"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != contract:
            raise ValueError(f"stale/incompatible {split} cache contract")
    else:
        meta_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    existing = _read_jsonl(feature_path, repair_partial_tail=True)
    index_by_key = {r.get("row_key"): r for r in selected}
    saved = {r["row_key"]: r for r in existing if r.get("row_key") in index_by_key}
    if len(saved) != len(existing):
        raise ValueError(f"{split} feature cache includes rows outside current selected data")
    if len(saved) == len(selected):
        return [saved[r["row_key"]] for r in selected]
    # Rebuild atomically if pilot limit was increased or cache has only a subset.
    done = set(saved)
    tokenizer.padding_side = "right"
    special_ids = getattr(tokenizer, "all_special_ids", [])
    start_time = time.monotonic()
    newly_completed = 0
    device = next(model.parameters()).device
    with feature_path.open("a", encoding="utf-8") as outf, progress_path.open("a", encoding="utf-8") as prog:
        for start in range(0, len(selected), batch_size):
            batch_indices = [i for i in range(start, min(len(selected), start + batch_size))
                             if selected[i]["row_key"] not in done]
            if not batch_indices:
                continue
            texts = [selected[i]["text"] for i in batch_indices]
            encoded = tokenizer(texts, truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt", add_special_tokens=True)
            if encoded["input_ids"].ndim != 2 or encoded["input_ids"].shape[1] == 0:
                raise ValueError("tokenizer produced a zero-width sequence")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            try:
                features = score_batch_chunked(model, ids, mask, special_ids)
                for idx, feat in zip(batch_indices, features):
                    obj = {"row_key": selected[idx]["row_key"], **feat}
                    outf.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
                    saved[obj["row_key"]] = obj
                newly_completed += len(features)
                outf.flush(); os.fsync(outf.fileno())
                elapsed = time.monotonic() - start_time
                completed = len(saved)
                rate = newly_completed / elapsed if elapsed else 0.0
                remain = max(0, len(selected) - completed)
                event = {"split": split, "completed": completed, "total": len(selected),
                         "batch_indices": batch_indices, "elapsed_seconds": elapsed,
                         "examples_per_second": rate,
                         "eta_seconds": remain / rate if rate else None,
                         "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if device.type == "cuda" else None,
                         "failed": False}
                prog.write(json.dumps(event, ensure_ascii=False) + "\n")
                prog.flush(); os.fsync(prog.fileno())
            except Exception as e:
                event = {"split": split, "batch_indices": batch_indices,
                         "failed": True, "error": repr(e)}
                prog.write(json.dumps(event, ensure_ascii=False) + "\n")
                prog.flush(); os.fsync(prog.fileno())
                raise
    return [saved[r["row_key"]] for r in selected]
