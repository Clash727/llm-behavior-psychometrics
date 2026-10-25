"""Token-aligned likelihood measurements under fixed text-prefix conditions."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

FEATURE_VERSION = "condition_probe_token_surprise_v1"
MODEL_NAME = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
BODY_BUDGET = 448
MAX_PREFIX_TOKENS = 64
CONDITIONS = ("raw", "neutral", "human_hypothesis", "machine_hypothesis")
TEMPLATES = {
    "raw": "",
    "neutral": "The following passage is ordinary written text:\n\n",
    "human_hypothesis": "The following passage is human written text:\n\n",
    "machine_hypothesis": "The following passage is machine written text:\n\n",
}
PAIRWISE = {
    "neutral_minus_raw": ("neutral", "raw"),
    "human_minus_neutral": ("human_hypothesis", "neutral"),
    "machine_minus_neutral": ("machine_hypothesis", "neutral"),
    "machine_minus_human": ("machine_hypothesis", "human_hypothesis"),
}
WINDOW = 16
LOGIT_CHUNK = 8


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_fingerprint(tokenizer) -> str:
    raw = json.loads(tokenizer.backend_tokenizer.to_str())
    raw.pop("padding", None)
    raw.pop("truncation", None)
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_row_key(split: str, index: int, text: str) -> str:
    # Match the established token_confidence row key exactly so CPU baselines
    # can join without translation; text_sha256 remains a separately stored ID.
    return hashlib.sha256(f"{split}\0{index}\0{text}".encode("utf-8")).hexdigest()


def read_rows(path: str | Path, split: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if not isinstance(obj.get("text"), str):
                raise ValueError("frozen rows require text strings")
            text_sha = hashlib.sha256(obj["text"].encode("utf-8")).hexdigest()
            rows.append({"text": obj["text"], "label": int(obj["label"]),
                         "source": obj.get("source", "unknown"), "original_index": len(rows),
                         "text_sha256": text_sha,
                         "row_key": stable_row_key(split, len(rows), obj["text"])})
    return rows


def encode_contract(tokenizer, templates: dict[str, str] = TEMPLATES,
                    body_budget: int = BODY_BUDGET) -> tuple[dict[str, list[int]], list[int]]:
    if set(templates) != set(CONDITIONS):
        raise ValueError("templates must define exactly the four fixed conditions")
    prefix_ids = {name: tokenizer.encode(text, add_special_tokens=False)
                  for name, text in templates.items()}
    if any(len(ids) > MAX_PREFIX_TOKENS for ids in prefix_ids.values()):
        raise ValueError("prefix exceeds 64 tokens")
    prompted_lengths = [len(prefix_ids[n]) for n in CONDITIONS[1:]]
    if len(set(prompted_lengths)) != 1:
        raise ValueError(f"neutral/human/machine prefix token lengths differ: {prompted_lengths}")
    specials = sorted(set(int(i) for i in getattr(tokenizer, "all_special_ids", ())))
    if body_budget != BODY_BUDGET:
        raise ValueError("condition probe body budget is fixed at 448 tokens")
    return prefix_ids, specials


def encode_body(tokenizer, text: str, *, body_budget: int = BODY_BUDGET) -> list[int]:
    # Tokenize the body on its own exactly once. No BOS/EOS or other additions.
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids[:body_budget]]


def surprises_from_logits(logits: torch.Tensor, body_ids: Sequence[Sequence[int]],
                          prefix_lengths: Sequence[int],
                          special_token_ids: Iterable[int] = ()) -> list[np.ndarray]:
    """Small reference scorer used for shift/mask checks and parity audits."""
    if logits.ndim != 3 or len(body_ids) != logits.shape[0] or len(prefix_lengths) != len(body_ids):
        raise ValueError("expected logits [B,T,V] and one body/prefix length per row")
    specials = set(int(x) for x in special_token_ids)
    results = []
    for i, (body, plen) in enumerate(zip(body_ids, prefix_lengths)):
        out = np.full(len(body), np.nan, dtype=np.float32)
        for j in range(1, len(body)):
            if int(body[j]) in specials:
                continue
            pos = int(plen) + j - 1
            if pos >= logits.shape[1]:
                raise ValueError("logits sequence is shorter than prefix plus body")
            row = logits[i, pos].float()
            out[j] = float(torch.logsumexp(row, dim=-1) - row[int(body[j])])
        results.append(out)
    return results


@torch.inference_mode()
def score_condition_batch(model, body_ids: Sequence[Sequence[int]],
                          prefix_ids: Sequence[int], *, special_token_ids: Iterable[int] = (),
                          logit_chunk: int = LOGIT_CHUNK,
                          device: torch.device | str | None = None) -> list[np.ndarray]:
    """Return body-offset-aligned FP32 surprises; offset zero is always NaN.

    Input IDs are exactly prefix_ids + body_ids. For body offset j>=1, hidden
    state at prefix_len+j-1 predicts body_ids[j]. This is the same target set in
    raw and prompted conditions; prefix tokens are context only.
    """
    if logit_chunk < 1:
        raise ValueError("logit_chunk must be positive")
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError("expected a causal LM exposing .model and .lm_head")
    if not body_ids:
        return []
    device = device or next(model.parameters()).device
    batch = len(body_ids)
    pref = [int(x) for x in prefix_ids]
    lengths = [len(x) for x in body_ids]
    width = max(len(pref) + n for n in lengths)
    if width == 0:
        return [np.full((0,), np.nan, dtype=np.float32) for _ in body_ids]
    ids = torch.zeros((batch, width), dtype=torch.long, device=device)
    mask = torch.zeros((batch, width), dtype=torch.long, device=device)
    for i, body in enumerate(body_ids):
        full = pref + [int(x) for x in body]
        ids[i, :len(full)] = torch.as_tensor(full, dtype=torch.long, device=device)
        mask[i, :len(full)] = 1
    output = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    hidden = output.last_hidden_state
    max_body = max(lengths, default=0)
    values = np.full((batch, max_body), np.nan, dtype=np.float32)
    specials = set(int(x) for x in special_token_ids)
    for start in range(1, max_body, logit_chunk):
        end = min(max_body, start + logit_chunk)
        offsets = torch.arange(start, end, device=device)
        target_hidden = torch.zeros((batch, end - start, hidden.shape[-1]),
                                    dtype=hidden.dtype, device=device)
        targets = torch.zeros((batch, end - start), dtype=torch.long, device=device)
        valid = torch.zeros((batch, end - start), dtype=torch.bool, device=device)
        for row, body in enumerate(body_ids):
            n = lengths[row]
            stop = min(end, n)
            if stop <= start:
                continue
            count = stop - start
            positions = offsets[:count] + len(pref) - 1
            target_hidden[row, :count] = hidden[row].index_select(0, positions)
            chunk_targets = torch.as_tensor(body[start:stop], dtype=torch.long, device=device)
            targets[row, :count] = chunk_targets
            valid[row, :count] = True
            if specials:
                for token_id in specials:
                    valid[row, :count] &= chunk_targets != token_id
        logits = model.lm_head(target_hidden)
        log_denom = torch.logsumexp(logits.float(), dim=-1)
        true_logits = logits.float().gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        surprise = log_denom - true_logits
        vals = surprise.detach().float().cpu().numpy()
        valid_np = valid.cpu().numpy()
        for row in range(batch):
            count = min(end, lengths[row]) - start
            if count > 0:
                valid_range = valid_np[row, :count]
                row_vals = values[row, start:start + count]
                row_vals[valid_range] = vals[row, :count][valid_range]
    return [values[i, :lengths[i]].copy() for i in range(batch)]


def summarize(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(x)
    v = x[valid].astype(np.float64)
    if not len(v):
        return {"scored_token_count": 0, "mean": None, "std": None,
                "q10": None, "q25": None, "median": None, "q75": None, "q90": None,
                "window16_means": []}
    windows = []
    for start in range(0, len(x), WINDOW):
        seg = x[start:start + WINDOW]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            windows.append({"start_body_offset": start, "end_body_offset_exclusive": min(len(x), start + WINDOW),
                            "mean": float(seg.mean(dtype=np.float64)), "count": int(len(seg))})
    q10, q25, med, q75, q90 = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"scored_token_count": int(len(v)), "mean": float(v.mean()),
            "std": float(v.std()), "q10": float(q10), "q25": float(q25),
            "median": float(med), "q75": float(q75), "q90": float(q90),
            "window16_means": windows}


def expected_scoring_mask(body_ids: Sequence[Sequence[int]], special_token_ids: Iterable[int],
                          width: int = BODY_BUDGET) -> np.ndarray:
    """Contract mask derived from token IDs, never from observed NaN/finite values."""
    specials = set(int(x) for x in special_token_ids)
    mask = np.zeros((len(body_ids), len(CONDITIONS), width), dtype=bool)
    for row, body in enumerate(body_ids):
        if len(body) > width:
            raise ValueError("body is longer than the fixed probe budget")
        for offset in range(1, len(body)):
            if int(body[offset]) not in specials:
                mask[row, :, offset] = True
    return mask


def validate_surprise_payload(surprises: np.ndarray, body_ids: Sequence[Sequence[int]],
                              special_token_ids: Iterable[int]) -> np.ndarray:
    expected = expected_scoring_mask(body_ids, special_token_ids, surprises.shape[-1])
    if surprises.shape != expected.shape:
        raise ValueError("surprise payload shape does not match the condition/body contract")
    if not np.isfinite(surprises[expected]).all():
        raise ValueError("non-finite surprise at an expected scored token")
    if not np.isnan(surprises[~expected]).all():
        raise ValueError("unexpected finite surprise outside the scoring mask")
    return expected


def build_row_record(row: dict[str, Any], surprises: np.ndarray,
                     body_length: int, prefix_lengths: Sequence[int]) -> dict[str, Any]:
    if surprises.shape != (len(CONDITIONS), BODY_BUDGET):
        raise ValueError("surprise array has wrong shape")
    conditions = {name: summarize(surprises[i, :body_length]) for i, name in enumerate(CONDITIONS)}
    deltas = {}
    for name, (left, right) in PAIRWISE.items():
        li, ri = CONDITIONS.index(left), CONDITIONS.index(right)
        diff = surprises[li, :body_length] - surprises[ri, :body_length]
        deltas[name] = summarize(diff)
    return {"row_key": row["row_key"], "text_sha256": row["text_sha256"],
            "original_index": int(row["original_index"]), "label": int(row["label"]),
            "source": str(row.get("source", "unknown")), "body_token_count": int(body_length),
            "conditions": conditions, "pairwise_surprise_deltas": deltas,
            "prefix_lengths": [int(x) for x in prefix_lengths]}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _process_start_ticks(pid: int | None = None) -> str | None:
    pid = pid or os.getpid()
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return raw[raw.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.payload = {"pid": os.getpid(), "process_start_ticks": _process_start_ticks(),
                        "started_utc": datetime.now(timezone.utc).isoformat()}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.payload, f); f.flush(); os.fsync(f.fileno())
                return self
            except FileExistsError:
                try:
                    old = json.loads(self.path.read_text())
                    pid = int(old["pid"])
                    alive = True
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        alive = False
                    same = alive and _process_start_ticks(pid) == old.get("process_start_ticks")
                    if same:
                        raise RuntimeError(f"condition probe run already locked by PID {pid}")
                    self.path.unlink()
                except (ValueError, KeyError, json.JSONDecodeError, OSError):
                    # A malformed lock is not presumed stale; avoid stealing it.
                    raise RuntimeError(f"cannot safely reclaim malformed lock {self.path}")
        raise RuntimeError(f"could not acquire run lock {self.path}")

    def __exit__(self, *_):
        try:
            old = json.loads(self.path.read_text())
            if old == self.payload:
                self.path.unlink()
        except OSError:
            pass


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("ab") as f:
        for row in records:
            f.write((json.dumps(row, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")) + "\n").encode("utf-8"))
        f.flush(); os.fsync(f.fileno())


def _read_jsonl(path: Path, repair_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    result, offset = [], 0
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if repair_tail and i == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as f:
                    f.truncate(offset); f.flush(); os.fsync(f.fileno())
                break
            raise ValueError(f"corrupt committed JSONL at {path}, byte {offset}") from exc
        offset += len(line)
    if repair_tail and raw and not raw.endswith(b"\n") and result:
        with path.open("ab") as f:
            f.write(b"\n"); f.flush(); os.fsync(f.fileno())
    return result


def validate_split_path(root: Path, split: str, path: Path) -> Path:
    if split not in ("train", "dev"):
        raise ValueError("only train and dev are allowed")
    expected = (root / f"artifacts/audited_train/splits/{split}.jsonl").resolve()
    actual = path.resolve()
    if actual != expected:
        raise ValueError(f"{split} input must be the frozen audited path {expected}")
    return expected


def make_manifest(*, split: str, input_path: Path, tokenizer, prefix_ids: dict[str, list[int]],
                  config_path: Path, templates: dict[str, str] = TEMPLATES) -> dict[str, Any]:
    import importlib.metadata
    code_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": FEATURE_VERSION, "split": split,
        "input_path": str(input_path), "input_sha256": sha256_file(input_path),
        "model": MODEL_NAME, "model_revision": MODEL_REVISION,
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "precision": "bfloat16", "attention_backend": "sdpa", "adapter": None,
        "templates": templates, "prefix_token_ids": prefix_ids,
        "prefix_lengths": {k: len(v) for k, v in prefix_ids.items()},
        "body_budget_tokens": BODY_BUDGET, "body_truncation": "right_truncate_independent_encode",
        "body_add_special_tokens": False, "input_contract": "prefix_ids + identical_body_ids",
        "scoring_batch_size": 1,
        "scoring_policy": "single_article_forward_per_condition_no_mixed_padding",
        "shard_rows": None,
        "scoring": "body offsets 1..N-1 only; exclude special targets; FP32 logsumexp minus true logit",
        "logit_chunk": LOGIT_CHUNK, "special_token_ids": sorted(set(int(x) for x in tokenizer.all_special_ids)),
        "feature_version": FEATURE_VERSION, "window_tokens": WINDOW,
        "config_sha256": sha256_file(config_path),
        "code_sha256": {"condition_probe.py": sha256_file(Path(__file__).resolve()),
                        "run_condition_probe.py": sha256_file(code_root / "scripts/run_condition_probe.py")},
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "numpy": np.__version__,
        "numerical_tolerance_note": "Each article-condition is a singleton forward; pilot repeats identical shapes in forward/reverse order with exact and 1e-6 checks",
    }


def _committed_shards(out: Path, manifest: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    entries = _read_jsonl(out / "shards.jsonl", repair_tail=True)
    found = {}
    for ent in entries:
        if ent.get("manifest_sha256") != hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest():
            raise ValueError("committed shard belongs to a different manifest")
        key = (int(ent["start"]), int(ent["end"]))
        path = out / ent["path"]
        if not path.exists() or sha256_file(path) != ent["sha256"]:
            raise ValueError(f"committed shard missing or hash mismatch: {path}")
        if key in found:
            raise ValueError("duplicate committed shard range")
        found[key] = ent
    return found


def _rebuild_rows_jsonl(out: Path, rows: list[dict[str, Any]], shards: dict[tuple[int, int], dict[str, Any]],
                        prefix_ids: dict[str, list[int]]) -> None:
    tmp = out / f"rows.jsonl.tmp-{os.getpid()}"
    expected = 0
    seen_original = set()
    with tmp.open("w", encoding="utf-8") as f:
        for (start, end), ent in sorted(shards.items()):
            if start != expected or end <= start:
                raise ValueError(f"committed shard coverage gap/overlap at selection index {expected}")
            with np.load(out / ent["path"], allow_pickle=False) as z:
                surprise = z["surprise"]
                lengths = z["body_length"]
                indexes = z["selection_index"]
                originals = z["original_index"]
                plen = z["prefix_lengths"]
                if indexes.tolist() != list(range(start, end)):
                    raise ValueError("shard selection indexes do not match committed range")
                batch_bodies = []
                for i, n in enumerate(lengths):
                    batch_bodies.append(z["body_token_ids"][i, :int(n)].tolist())
                expected_mask = expected_scoring_mask(batch_bodies,
                    json.loads((out / "manifest.json").read_text())["special_token_ids"], BODY_BUDGET)
                if not np.array_equal(z["valid_mask"], expected_mask):
                    raise ValueError("shard valid mask differs from the body-token scoring contract")
                if not np.isfinite(surprise[expected_mask]).all() or not np.isnan(surprise[~expected_mask]).all():
                    raise ValueError("shard surprises violate finite-on-mask contract")
                for i, index in enumerate(indexes):
                    index = int(index)
                    original = int(originals[i])
                    if original != int(rows[index]["original_index"]):
                        raise ValueError("shard original index does not align with selected row")
                    if original in seen_original:
                        raise ValueError("duplicate original row among committed shards")
                    seen_original.add(original)
                    record = build_row_record(rows[index], surprise[i], int(lengths[i]), plen[i])
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                       separators=(",", ":")) + "\n")
            expected = end
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out / "rows.jsonl")


@torch.inference_mode()
def extract_split(*, rows: list[dict[str, Any]], split: str, out: Path, tokenizer, model,
                  manifest: dict[str, Any], prefix_ids: dict[str, list[int]],
                  shard_rows: int = 1, scoring_batch_size: int = 1,
                  limit: int | None = None) -> dict[str, Any]:
    if split not in ("train", "dev") or shard_rows < 1 or scoring_batch_size != 1:
        raise ValueError("condition probe requires valid split, positive shard_rows, and singleton scoring")
    if manifest.get("scoring_batch_size") != 1 or manifest.get("shard_rows") != shard_rows:
        raise ValueError("manifest must bind singleton scoring and the configured shard row count")
    selected = rows[:limit] if limit is not None else rows
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("existing run manifest differs; refusing to overwrite artifacts")
    else:
        if any(out.iterdir()):
            raise ValueError("output directory contains unmanifested artifacts")
        atomic_json(manifest_path, manifest)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    shard_root = out / "shards" / split
    shard_root.mkdir(parents=True, exist_ok=True)
    with RunLock(out / "run.lock"):
        committed = _committed_shards(out, manifest)
        done = set()
        for (start, end), ent in committed.items():
            if end > len(selected) or end <= start:
                raise ValueError("committed shard range is outside selected rows")
            with np.load(out / ent["path"], allow_pickle=False) as z:
                if z["selection_index"].tolist() != list(range(start, end)):
                    raise ValueError("shard original-index sequence mismatch")
                done.update(range(start, end))
        # Uncommitted temporary and orphan NPZ payloads never become visible.
        for p in shard_root.iterdir():
            if ".tmp-" in p.name or p.suffix == ".npz" and p.relative_to(out).as_posix() not in {e["path"] for e in committed.values()}:
                p.unlink()
        start_time = time.monotonic()
        completed = len(done)
        newly_completed = 0
        progress_path = out / "progress.jsonl"
        for start in range(0, len(selected), shard_rows):
            end = min(len(selected), start + shard_rows)
            if all(i in done for i in range(start, end)):
                continue
            if any(i in done for i in range(start, end)):
                raise ValueError("partial committed batch range cannot be resumed")
            batch = selected[start:end]
            bodies = [encode_body(tokenizer, r["text"]) for r in batch]
            surprises = np.full((len(batch), len(CONDITIONS), BODY_BUDGET), np.nan, dtype=np.float32)
            device = next(model.parameters()).device
            peak = 0
            t_batch = time.monotonic()
            try:
                # Scientific contract: each article is a separate forward pass,
                # even when several rows are stored in one atomic NPZ shard.
                for ri, body in enumerate(bodies):
                    for ci, condition in enumerate(CONDITIONS):
                        scored = score_condition_batch(model, [body], prefix_ids[condition],
                                                       special_token_ids=tokenizer.all_special_ids,
                                                       device=device)
                        surprises[ri, ci, :len(body)] = scored[0]
                        if device.type == "cuda":
                            peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
                lengths = np.asarray([len(x) for x in bodies], dtype=np.int32)
                max_len = max(lengths.tolist(), default=0)
                token_ids = np.zeros((len(batch), BODY_BUDGET), dtype=np.int32)
                for i, body in enumerate(bodies):
                    token_ids[i, :len(body)] = np.asarray(body, dtype=np.int32)
                expected_mask = expected_scoring_mask(bodies, tokenizer.all_special_ids, BODY_BUDGET)
                valid = expected_mask
                if not np.isfinite(surprises[expected_mask]).all() or not np.isnan(surprises[~expected_mask]).all():
                    raise ValueError("computed surprises violate finite-on-mask contract")
                path = shard_root / f"shard-{start:08d}-{end:08d}.npz"
                _atomic_npz(path, surprise=surprises, body_token_ids=token_ids,
                            valid_mask=valid, body_length=lengths,
                            prefix_lengths=np.tile(np.asarray([len(prefix_ids[x]) for x in CONDITIONS], dtype=np.int16), (len(batch), 1)),
                            original_index=np.asarray([int(r["original_index"]) for r in batch], dtype=np.int32),
                            selection_index=np.arange(start, end, dtype=np.int32))
                # Reload before commit; validate dimensions and finite-on-mask contract.
                with np.load(path, allow_pickle=False) as z:
                    if z["surprise"].shape != (len(batch), 4, BODY_BUDGET) or not np.isfinite(z["surprise"][z["valid_mask"]]).all():
                        raise ValueError("serialized shard validation failed")
                    if not np.array_equal(z["valid_mask"], expected_mask) or not np.isnan(z["surprise"][~expected_mask]).all():
                        raise ValueError("serialized mask/NaN contract failed")
                    if not np.array_equal(z["body_token_ids"][:, :max_len], token_ids[:, :max_len]):
                        raise ValueError("serialized body token IDs changed")
                ent = {"split": split, "start": start, "end": end,
                       "path": path.relative_to(out).as_posix(), "sha256": sha256_file(path),
                       "manifest_sha256": manifest_hash}
                _append_jsonl(out / "shards.jsonl", [ent])
                committed[(start, end)] = ent
                done.update(range(start, end)); completed += len(batch)
                newly_completed += len(batch)
                rate = newly_completed / max(time.monotonic() - start_time, 1e-9)
                atomic_json(out / "run_status.json", {
                    "pid": os.getpid(), "process_start_ticks": _process_start_ticks(),
                    "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running",
                    "split": split, "last_batch": [start, end], "completed": completed,
                    "total": len(selected), "elapsed_seconds": time.monotonic() - start_time,
                    "examples_per_second": rate, "reused_rows": len(selected) - newly_completed,
                    "eta_seconds": (len(selected) - completed) / rate if rate else None,
                    "batch_seconds": time.monotonic() - t_batch,
                    "gpu_peak_allocated_bytes": peak if device.type == "cuda" else None,
                    "failed": False})
                _append_jsonl(progress_path, [{"split": split, "start": start, "end": end,
                                               "completed": completed, "total": len(selected),
                                               "examples_per_second": rate,
                                               "eta_seconds": (len(selected) - completed) / rate if rate else None,
                                               "gpu_peak_allocated_bytes": peak if device.type == "cuda" else None,
                                               "failed": False}])
            except Exception as exc:
                atomic_json(out / "run_status.json", {"pid": os.getpid(),
                    "process_start_ticks": _process_start_ticks(), "state": "failed",
                    "split": split, "last_batch": [start, end], "completed": completed,
                    "total": len(selected), "failed": True, "error": repr(exc),
                    "gpu_peak_allocated_bytes": peak})
                _append_jsonl(progress_path, [{"split": split, "start": start, "end": end,
                                               "completed": completed, "total": len(selected),
                                               "failed": True, "error": repr(exc),
                                               "gpu_peak_allocated_bytes": peak}])
                raise
        _rebuild_rows_jsonl(out, selected, committed, prefix_ids)
        final = {"state": "complete", "split": split, "completed": len(selected),
                 "total": len(selected), "elapsed_seconds": time.monotonic() - start_time,
                 "examples_per_second": newly_completed / max(time.monotonic() - start_time, 1e-9) if newly_completed else None,
                 "newly_completed_rows": newly_completed, "reused_rows": len(selected) - newly_completed,
                 "gpu_peak_allocated_bytes": max((int(x.get("gpu_peak_allocated_bytes") or 0)
                                                     for x in _read_jsonl(progress_path)), default=0),
                 "pid": os.getpid(), "process_start_ticks": _process_start_ticks(),
                 "failed": False}
        atomic_json(out / "run_status.json", final)
    return final
